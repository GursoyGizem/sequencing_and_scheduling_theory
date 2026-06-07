
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

    # Sag Kaydirma Stratejisi:
    # Islerin ardindan gelen bosluklari kapatmak icin, her asamadaki isleri saga kaydir.
    # Boylece bekleme enerjisi azalir, Cmax kesinlikle artmaz.

    # Temel kural (makine penceresi buyumez):
    #   - Son is KAYDIRILMAZ (son is kayarsa pencere genisler, TEC/TWC artar).
    #   - Yalnizca "saga dogru bosluk" olan isler kaydırılır.
    #   - Kaydirma miktarı: isin sagındaki komsuya yapisana kadar.
    #   - Oncel kisit (C[(j,k-1)]) ve sonraki asama kisiti (S[(j,k+1)])
    #     asilmaz; Cmax kesinlikle artmaz.
    #
    # Yon: son asamadan ilk asamaya (k = SN-1 -> 0).
    #      Boylece k+1 asamasinda yapılan guncellemeler
    #      k asamasinin kisitlarini guncel tutar.

    for f in range(FN):
        isler = fabrika_isler[f]
        if not isler:
            continue

        for k in range(SN - 1, -1, -1):
            for m in range(inst.MN[f][k]):
                # Makineye atanan isler, mevcut baslama zamanina gore sirali
                jobs_m = sorted(
                    [j for j in isler if MA[j][k] % inst.MN[f][k] == m],
                    key=lambda j: S[(j, k)]
                )
                if len(jobs_m) < 2:
                    continue   # tek is varsa bosluk olmaz

                # Sagdan sola isle; son is kaydirilmaz, oncesi sagdaki komsuya kadar kayar
                for idx in range(len(jobs_m) - 2, -1, -1):
                    j     = jobs_m[idx]
                    j_sag = jobs_m[idx + 1]
                    at_j  = AT[(j, k)]

                    # Sagdaki komsu ile bosluk var mi?
                    bosluk = S[(j_sag, k)] - C[(j, k)]
                    if bosluk <= 1e-9:
                        continue   # zaten bosluk yok

                    # En gec baslama: sagdaki is baslamadan hemen once bitis
                    en_gec = S[(j_sag, k)] - at_j

                    # Sonraki asama kisiti: j'nin k+1 asamasi gecikmemeli
                    if k < SN - 1:
                        en_gec = min(en_gec, S[(j, k + 1)] - at_j)

                    # Oncel kisit: k-1 asamasi bitmeden baslayamaz
                    en_erken = C[(j, k - 1)] if k > 0 else 0.0

                    # Gecerli ve sag-kaydirma mumkun mu?
                    if en_gec > S[(j, k)] + 1e-9 and en_gec >= en_erken - 1e-9:
                        yeni_bas = max(en_gec, en_erken)
                        S[(j, k)] = yeni_bas
                        C[(j, k)] = yeni_bas + at_j

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
                isler_sirali = sorted(isler_m, key=lambda x: x[1])  # baslama zamanina gore sirala
                bekleme = 0.0
                for idx in range(len(isler_sirali) - 1):
                    mevcut_bitis = isler_sirali[idx][2]
                    sonraki_baslama = isler_sirali[idx + 1][1]
                    bosluk = sonraki_baslama - mevcut_bitis
                    bekleme += max(0.0, bosluk)
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