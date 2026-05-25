import numpy as np
import random
from copy import deepcopy

from solution import Cozum, rastgele_cozum, cozumle
from pareto import baskin_olmayanlar, arsiv_guncelle, pddr_ff, baskin_mi
from local_search import (fabrikalar_arasi_yerel_arama, q_ogrenme_fabrika_ici_arama, vns_operatoru_uygula)

"""
    Surumu 4 alt suriye bol:
    G1: Cmax sinirina odakli (15 parcacik)
    G2: TEC sinirina odakli  (15 parcacik)
    G3: TWC sinirina odakli  (15 parcacik)
    G4: Merkez denge bolgesi (55 parcacik)

    Sinir alt suruleri: ilgili hedefte en iyi parcaciklardan olusur.
    Merkez alt suru: PDDR-FF'ye gore en guclu parcaciklardan olusur.
"""
def suru_ayristir(suru, gbest_set, boyutlar=(15, 15, 15, 55)):
    n1, n2, n3, n4 = boyutlar
    N = len(suru)

    # Her parcacigin hedef degerleri
    gec = [c for c in suru if c.hedefler is not None]
    if not gec:
        return [suru[:n1], suru[n1:n1+n2],
                suru[n1+n2:n1+n2+n3], suru[n1+n2+n3:]]

    # G1: Cmax'i en kucuk olanlar
    sirali_cmax = sorted(gec, key=lambda c: c.hedefler[0])
    G1 = sirali_cmax[:n1]

    # G2: TEC'i en kucuk olanlar
    sirali_tec = sorted(gec, key=lambda c: c.hedefler[1])
    G2 = sirali_tec[:n2]

    # G3: TWC'yi en kucuk olanlar
    sirali_twc = sorted(gec, key=lambda c: c.hedefler[2])
    G3 = sirali_twc[:n3]

    # G4: PDDR-FF degeri en kucuk olanlar (en guclu cozumler)
    arsiv = gbest_set if gbest_set else gec
    pddr_degerler = [(c, pddr_ff(c, arsiv)) for c in gec]
    sirali_pddr = sorted(pddr_degerler, key=lambda x: x[1])
    G4 = [c for c, _ in sirali_pddr[:n4]]

    return G1, G2, G3, G4

"""
PSO Guncelleme
JS vektoru icin degisim dizisi operatoru.
js_x ile js_y arasindaki farki hesapla, r oraninda uygula.
"""

def js_degisim_dizisi(js_x, js_y, r):
    js_yeni = js_x.copy()
    n = len(js_x)

    ciftler = []
    gecici = js_x.copy()
    for i in range(n):
        if gecici[i] != js_y[i]:
            j = gecici.index(js_y[i])
            ciftler.append((i, j))
            gecici[i], gecici[j] = gecici[j], gecici[i]

    uygulanan = int(max(1, round(r * len(ciftler)))) if ciftler else 0
    for i, j in ciftler[:uygulanan]:
        idx_i = js_yeni.index(js_x[i]) if js_x[i] in js_yeni else i
        idx_j = js_yeni.index(js_x[j]) if js_x[j] in js_yeni else j
        js_yeni[idx_i], js_yeni[idx_j] = js_yeni[idx_j], js_yeni[idx_i]

    return js_yeni

""" FA, MA, WA vektorleri icin caprazlama ve mutasyon. """
def caprazlama_mutasyon(vektor_x, vektor_y, capraz_oran, mutasyon_oran, min_val=0, max_val_list=None):
    n = len(vektor_x)
    yeni = vektor_x.copy()

    # Iki nokta caprazlama
    if random.random() < capraz_oran and n >= 2:
        p1, p2 = sorted(random.sample(range(n), 2))
        yeni[p1:p2+1] = vektor_y[p1:p2+1]

    # Mutasyon: rastgele bir konumu degistir
    if random.random() < mutasyon_oran:
        pos = random.randint(0, n-1)
        if max_val_list and pos < len(max_val_list):
            yeni[pos] = random.randint(0, max(0, max_val_list[pos]-1))

    return yeni

"""
    Algoritma 2: Kuresel Arama Guncelleme
    Adim 1: Rastgele kesif
    Adim 2: pbest'e yonelme
    Adim 3: gbest'e yonelme
"""
def parcacik_guncelle(parcacik, pbest, gbest, inst, params):
    r1 = params.get('r1', 0.4)
    r2 = params.get('r2', 0.4)
    cv2 = params.get('cv2', 0.2)
    cv3 = params.get('cv3', 0.4)
    cv4 = params.get('cv4', 0.3)
    mv2 = params.get('mv2', 0.06)
    mv3 = params.get('mv3', 0.10)
    mv4 = params.get('mv4', 0.25)

    JN, SN, FN = inst.JN, inst.SN, inst.FN
    yeni = parcacik.kopyala()

    # Adim 1: rastgele kesif
    js_karisik = yeni.JS.copy()
    random.shuffle(js_karisik)
    yeni.JS = js_degisim_dizisi(yeni.JS, js_karisik, 0.1)

    # Adim 2: pbest'e yonelme
    if pbest.hedefler:
        yeni.JS = js_degisim_dizisi(yeni.JS, pbest.JS, r1)

    # Adim 3: gbest'e yonelme
    if gbest and gbest.hedefler:
        yeni.JS = js_degisim_dizisi(yeni.JS, gbest.JS, r2)

    # FA Guncelleme (Caprazlama + Mutasyon) 
    for ebeveyn in [pbest, gbest]:
        if ebeveyn is None or not ebeveyn.hedefler:
            continue
        yeni.FA = caprazlama_mutasyon(
            yeni.FA, ebeveyn.FA, cv2, mv2,
            max_val_list=[FN]*JN
        )

    # MA Guncelleme 
    for ebeveyn in [pbest, gbest]:
        if ebeveyn is None or not ebeveyn.hedefler:
            continue
        for j in range(JN):
            f = yeni.FA[j]
            maks = [inst.MN[f][k] for k in range(SN)]
            satir_yeni = caprazlama_mutasyon(
                yeni.MA[j], ebeveyn.MA[j], cv3, mv3,
                max_val_list=maks
            )
            # Gecerlilik: makine fabrikada var mi?
            for k in range(SN):
                satir_yeni[k] = satir_yeni[k] % inst.MN[f][k]
            yeni.MA[j] = satir_yeni

    # WA Guncelleme
    for ebeveyn in [pbest, gbest]:
        if ebeveyn is None or not ebeveyn.hedefler:
            continue
        for f in range(FN):
            wn = inst.WN[f]
            yeni.WA[f] = caprazlama_mutasyon(
                yeni.WA[f], ebeveyn.WA[f], cv4, mv4,
                max_val_list=[wn]*wn
            )

    return yeni

"""
Algoritma 1: HMOPSO-QLS Ana Cercevesi

    inst   : DHHFSPInstance
    N      : Toplam parcacik sayisi
    E_max  : Maksimum degerlendirme sayisi
    params : Algoritma parametreleri
    verbose: Ilerleme yazdir
"""
def hmopso_qls(inst, N=100, E_max=5000, params=None, verbose=True):
    if params is None:
        params = {
            'r1': 0.4, 'r2': 0.4,
            'cv2': 0.2, 'cv3': 0.4, 'cv4': 0.3,
            'mv2': 0.06, 'mv3': 0.10, 'mv4': 0.25,
            'boyutlar': (15, 15, 15, 55),
            'Q_Times': 40, 'L_Times': 40,
            'gamma': 0.8, 'epsilon': 0.9, 'alpha': 0.1
        }

    boyutlar = params['boyutlar']
    if sum(boyutlar) > N:
        oran = N / sum(boyutlar)
        boyutlar = tuple(max(1, int(b * oran)) for b in boyutlar)
        boyutlar = boyutlar[:3] + (N - sum(boyutlar[:3]),)
        params['boyutlar'] = boyutlar

    t = 0
    E_cur = 0

    print(f"Ornek: {inst.JN} is, {inst.SN} asama, {inst.FN} fabrika")
    print(f"Parcacik: {N}, Max Degerlendirme: {E_max}")


    # Baslangic surumu olustur
    suru = []
    for _ in range(N):
        c = rastgele_cozum(inst)
        cozumle(c, inst)
        suru.append(c)
    E_cur += N

    # pbest ve gbest baslat
    pbest_set = [c.kopyala() for c in suru]
    gbest_set = baskin_olmayanlar(suru)

    # Yakinlasma takibi
    gecmis_cmax = []
    gecmis_tec = []
    gecmis_twc = []
    gecmis_pareto_boyut = []

    while E_cur < E_max:
        t += 1
        # Ayristirma: Surumu 4 alt suruye bol
        G1, G2, G3, G4 = suru_ayristir(suru, gbest_set, params['boyutlar'])
        alt_surular = [G1, G2, G3, G4]

        # Kuresel Arama (PSO)
        # Her alt suru icin ilgili hedefe gore en iyi gbest sec
        def gbest_sec(gs_no):
            if not gbest_set:
                return min(suru, key=lambda c: c.hedefler[min(gs_no,2)]
                           if c.hedefler else float('inf'))
            if gs_no == 0:   # G1: Cmax en kucuk
                return min(gbest_set, key=lambda c: c.hedefler[0])
            elif gs_no == 1: # G2: TEC en kucuk
                return min(gbest_set, key=lambda c: c.hedefler[1])
            elif gs_no == 2: # G3: TWC en kucuk
                return min(gbest_set, key=lambda c: c.hedefler[2])
            else:            # G4: PDDR-FF en kucuk
                return min(gbest_set, key=lambda c: pddr_ff(c, gbest_set))

        yeni_suru = []
        gs_no_listesi = []   # her parcacigin hangi alt suruye ait oldugu
        for gs_no, gs in enumerate(alt_surular):
            gbest_ornek = gbest_sec(gs_no)
            for parcacik in gs:
                pbest_idx = suru.index(parcacik) if parcacik in suru else 0
                pbest = pbest_set[pbest_idx]
                guncellenmis = parcacik_guncelle(
                    parcacik, pbest, gbest_ornek, inst, params)
                cozumle(guncellenmis, inst)
                yeni_suru.append(guncellenmis)
                gs_no_listesi.append(gs_no)
        E_cur += len(yeni_suru)

        # Fabrikalar Arasi Yerel Arama — PSO ciktisi yeni_suru uzerinde calistir
        if_suru = []
        for parcacik, gs_no in zip(yeni_suru, gs_no_listesi):
            guncellenmis_if = fabrikalar_arasi_yerel_arama(
                parcacik, inst, gs_no, gbest_set)
            if_suru.append(guncellenmis_if)
        E_cur += len(if_suru)

        # Q-Ogrenme Tabanli Fabrika Ici Arama — if_suru uzerinde calistir
        in_suru = []
        for parcacik, gs_no in zip(if_suru, gs_no_listesi):
            guncellenmis_in = q_ogrenme_fabrika_ici_arama(
                parcacik, inst, gs_no, gbest_set,
                Q_Times=params['Q_Times'],
                L_Times=params['L_Times'],
                alpha=params['alpha'],
                gamma=params['gamma'],
                epsilon=params['epsilon']
            )
            in_suru.append(guncellenmis_in)
        E_cur += len(in_suru)

        # Birlesim ve PDDR-FF secimi — orijinal suru + tam guncel in_suru
        R_t = suru + in_suru
        R_t = [c for c in R_t if c.hedefler is not None]

        # PDDR-FF ile en iyi N parcaciği sec
        aralik = gbest_set if gbest_set else R_t
        pddr_degerler = [(c, pddr_ff(c, aralik)) for c in R_t]
        sirali = sorted(pddr_degerler, key=lambda x: x[1])
        suru = [c for c, _ in sirali[:N]]

        # pbest_set ve gbest_set guncelle 
        for i, c in enumerate(suru):
            if i < len(pbest_set):
                pb = pbest_set[i]
                if (pb.hedefler is None or
                        baskin_mi(c.hedefler, pb.hedefler)):
                    pbest_set[i] = c.kopyala()
            else:
                pbest_set.append(c.kopyala())

        gbest_set = arsiv_guncelle(gbest_set, suru)

        if gbest_set:
            cmax_vals = [c.hedefler[0] for c in gbest_set]
            tec_vals  = [c.hedefler[1] for c in gbest_set]
            twc_vals  = [c.hedefler[2] for c in gbest_set]
            gecmis_cmax.append(min(cmax_vals))
            gecmis_tec.append(min(tec_vals))
            gecmis_twc.append(min(twc_vals))
            gecmis_pareto_boyut.append(len(gbest_set))

        if verbose and t % 10 == 0:
            print(f" Iter {t:4d} | E_cur={E_cur:6d} | "
                  f"Pareto={len(gbest_set):3d} | "
                  f"C_max={gecmis_cmax[-1]:.2f} | "
                  f"TEC={gecmis_tec[-1]:.2f} | "
                  f"TWC={gecmis_twc[-1]:.2f}")

    print(f"Tamamlandi: {t} iterasyon, {E_cur} degerlendirme")
    print(f"Pareto cephesi boyutu: {len(gbest_set)}")

    gecmis = {
        'cmax': gecmis_cmax,
        'tec': gecmis_tec,
        'twc': gecmis_twc,
        'pareto_boyut': gecmis_pareto_boyut
    }

    return gbest_set, gecmis
