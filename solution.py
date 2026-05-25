
"""
Encoding, Decoding ve Hedef Fonksiyon Hesaplama dosyasi. DHHFSP-WME cozumlerini 4 vektorle temsil eder:
JS : Is siralama vektoru (islerin islenme sirasini belirten permutasyon)
FA : Fabrika atama vektoru (her isin hangi fabrikada islenecegini belirten vektor)
MA : Makine atama vektoru (her isin her asamasinda hangi makinede islenecegini belirten matris)
WA : Isci atama vektoru (her fabrikada islerin hangi iscilerce yapilacagini belirten vektor)

4-vektor cozumu gercek bir uretime esle;
C_max, TEC, TWC degerlerini hesapla.

Kuralllar:
    - Asama 1: JS sirasiyla isle
    - Asama k>1: ECT (En Erken Tamamlanan Once) sirasi
    - Makine secimi: MA vektorunden
    - Isci secimi: WA vektorunden (makine -> isci eslemesi)
    - Sag kaydirma: bekleme enerjisini azalt
"""
import numpy as np
from copy import deepcopy

class Cozum:
    def __init__(self, JS, FA, MA, WA):
        self.JS = JS.copy()
        self.FA = FA.copy()
        self.MA = [row.copy() for row in MA]
        self.WA = [w.copy() for w in WA]
        self.hedefler = None   # (C_max, TEC, TWC)
        self.eval_degeri = None

    def kopyala(self):
        yeni = Cozum(self.JS, self.FA, self.MA, self.WA)
        yeni.hedefler = self.hedefler
        yeni.eval_degeri = self.eval_degeri
        return yeni


def rastgele_cozum(inst):
    JN, SN, FN = inst.JN, inst.SN, inst.FN

    JS = list(np.random.permutation(JN))
    FA = list(np.random.randint(0, FN, size=JN))

    MA = []
    for j in range(JN):
        satir = []
        for k in range(SN):
            f = FA[j]
            makine_sayisi = inst.MN[f][k]
            satir.append(np.random.randint(0, makine_sayisi))
        MA.append(satir)

    # WA: fabrika basi isci atamasi - permutasyon
    WA = []
    for f in range(FN):
        toplam_makine = inst.WN[f]
        WA.append(list(np.random.permutation(toplam_makine)))

    return Cozum(JS, FA, MA, WA)


def cozumle(cozum, inst):
    JN, SN, FN = inst.JN, inst.SN, inst.FN
    JS, FA, MA, WA = cozum.JS, cozum.FA, cozum.MA, cozum.WA

    # Hedef fonksiyon degerlerini hesapla: C_max, TEC, TWC
    # f fabrikası, k asaması, m makinesindeki isciyi bulmak icin:
    def isci_bul(f, k, m):
        idx = sum(inst.MN[f][kk] for kk in range(k)) + m
        return WA[f][idx % inst.WN[f]]

    S = {}   # S[(j,k)] = baslama zamani
    C = {}   # C[(j,k)] = bitis zamani
    AT = {}  # AT[(j,k)] = gercek islem suresi

    # Fabrika basi makine musaitlik zamanlari
    mak_mevcut = []
    for f in range(FN):
        asama_mevcut = [[0.0] * inst.MN[f][k] for k in range(SN)]
        mak_mevcut.append(asama_mevcut)

    # Her fabrikaya atanan isler (JS sirasini koru)
    fabrika_isler = [[] for _ in range(FN)]
    for j in JS:
        fabrika_isler[FA[j]].append(j)

    for f in range(FN):
        isler = fabrika_isler[f]
        if not isler:
            continue

        for k in range(SN):
            if k == 0:
                siralama = isler  # JS sirasi
            else:
                # ECT: onceki asamayi en erken bitiren once
                siralama = sorted(isler, key=lambda j: C[(j, k-1)])

            for j in siralama:
                m = MA[j][k]
                # Gecerlilik kontrolu: makine bu fabrikada var mi?
                m = m % inst.MN[f][k]
                w = isci_bul(f, k, m)

                # Gercek islem suresi = ST / (Makine hizi x Isci verimliligi)
                at = inst.ST[j][k] / (inst.MS[f][k][m] * inst.WE[f][w])
                AT[(j, k)] = at

                if k == 0:
                    en_erken = 0.0
                else:
                    en_erken = C[(j, k-1)]  # Onceki asama bitisi

                baslama = max(en_erken, mak_mevcut[f][k][m])
                S[(j, k)] = baslama
                C[(j, k)] = baslama + at
                mak_mevcut[f][k][m] = C[(j, k)]

    # --- SAG KAYDIRMA STRATEJISI ---
    # Her makinedeki isleri mumkun oldugunca saga kaydirarak TEC ve TWC'yi azalt.
    # Kural: Cmax artmamali, hicbir isin sonraki asamasi gecikmemeli.
    # Yon: son asamadan ilk asamaya (geriye dogru) isle — boylece her asama,
    #      sonraki asamanin guncel baslama zamani kisitini kullanir.

    # Once gecici Cmax hesapla (kaydirma siniri)
    Cmax_gecici = max(C[(j, SN - 1)] for j in range(JN))

    for f in range(FN):
        isler = fabrika_isler[f]
        if not isler:
            continue

        for k in range(SN - 1, -1, -1):           # son asamadan geriye
            for m in range(inst.MN[f][k]):
                # Bu makinedeki isler, mevcut baslama zamanina gore sirali
                jobs_m = sorted(
                    [j for j in isler if MA[j][k] % inst.MN[f][k] == m],
                    key=lambda j: S[(j, k)]
                )
                if not jobs_m:
                    continue

                # Sagdan sola isle: son is -> ilk is
                for idx in range(len(jobs_m) - 1, -1, -1):
                    j = jobs_m[idx]
                    at_j = AT[(j, k)]

                    # Izin verilen en gec bitis:
                    if k == SN - 1:
                        # Son asama: Cmax_gecici'yi gecme
                        ust_sinir = Cmax_gecici
                    else:
                        # Ara asama: sonraki asamanin (guncel) baslama zamanini gecme
                        ust_sinir = S[(j, k + 1)]

                    # Ayni makinede sagdaki komsu is ile cakisma olmasin
                    if idx < len(jobs_m) - 1:
                        j_sag = jobs_m[idx + 1]
                        ust_sinir = min(ust_sinir, S[(j_sag, k)])

                    # En gec izin verilen baslama zamani
                    en_gec = ust_sinir - at_j

                    # En erken izin verilen baslama zamani (oncel kisit)
                    en_erken = C[(j, k - 1)] if k > 0 else 0.0

                    # Sadece saga kaydirma: en_gec >= en_erken ise kaydirma yap
                    if en_gec >= en_erken + 1e-9 and en_gec > S[(j, k)] + 1e-9:
                        S[(j, k)] = en_gec
                        C[(j, k)] = en_gec + at_j

    Cmax = max(C[(j, SN - 1)] for j in range(JN))
    TEC = 0.0
    for f in range(FN):
        for k in range(SN):
            for m in range(inst.MN[f][k]):
                isler_m = [(j, S[(j,k)], C[(j,k)])
                           for j in fabrika_isler[f]
                           if MA[j][k] % inst.MN[f][k] == m]
                if not isler_m:
                    continue

                # Isleme enerjisi
                for j, s, c in isler_m:
                    TEC += inst.MECPU[f][k][m] * AT[(j,k)]

                # Bekleme enerjisi
                ilk_bas = min(s for _, s, c in isler_m)
                son_bit = max(c for _, s, c in isler_m)
                toplam_islem = sum(AT[(j,k)] for j, _, _ in isler_m)
                bekleme = max(0, (son_bit - ilk_bas) - toplam_islem)
                TEC += inst.MECSU[f][k][m] * bekleme

    TWC = 0.0
    for f in range(FN):
        for k in range(SN):
            for m in range(inst.MN[f][k]):
                isler_m = [(j, S[(j,k)], C[(j,k)])
                           for j in fabrika_isler[f]
                           if MA[j][k] % inst.MN[f][k] == m]
                if not isler_m:
                    continue

                w = isci_bul(f, k, m)
                ilk_bas = min(s for _, s, c in isler_m)
                son_bit = max(c for _, s, c in isler_m)
                sure = son_bit - ilk_bas
                TWC += inst.WCPU[f][w] * sure

    cozum.hedefler = (Cmax, TEC, TWC)
    return Cmax, TEC, TWC