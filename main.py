import os
import time
import numpy as np
import matplotlib

from problem     import DHHFSPInstance, kucuk_ornek
from solution      import rastgele_cozum, cozumle
from hmopso_qls  import hmopso_qls
from visualization import (pareto_cephesi_ciz, yakinlasma_grafigi, hedef_istatistikleri_yazdir, gantt_grafigi)

matplotlib.use('Agg')

CIKTI = os.path.join(os.path.dirname(__file__), 'sonuclar')
os.makedirs(CIKTI, exist_ok=True)

def sonuc_tablosu_yazdir(etiket, gbest_set, sure):
    hedefler = [c.hedefler for c in gbest_set if c.hedefler]
    if not hedefler:
        print("Sonuc yok")
        return

    cmax_vals = [h[0] for h in hedefler]
    tec_vals  = [h[1] for h in hedefler]
    twc_vals  = [h[2] for h in hedefler]

    print(f"Deney: {etiket}")
    print(f"Calisma suresi      : {sure:.2f} sn")
    print(f"Pareto cozum sayisi : {len(gbest_set)}")
    print(f"{'Hedef':<10} {'Min':>10} {'Max':>10} {'Ort':>10} {'Std':>10}")

    for isim, vals in [('C_max', cmax_vals), ('TEC', tec_vals), ('TWC', twc_vals)]:
        print(f"{isim:<10} "
              f"{min(vals):>10.3f} "
              f"{max(vals):>10.3f} "
              f"{np.mean(vals):>10.3f} "
              f"{np.std(vals):>10.3f}")

def deney_kucuk():
    inst = kucuk_ornek()
    inst.yazdir()

    params = {
        'r1': 0.4, 'r2': 0.4,
        'cv2': 0.2, 'cv3': 0.4, 'cv4': 0.3,
        'mv2': 0.06, 'mv3': 0.10, 'mv4': 0.25,
        'boyutlar': (5, 5, 5, 15),   
        'Q_Times': 20, 'L_Times': 20,
        'gamma': 0.8, 'epsilon': 0.9, 'alpha': 0.1
    }

    t0 = time.time()
    gbest_set, gecmis = hmopso_qls(
        inst, N=30, E_max=3000, params=params, verbose=True)
    sure = time.time() - t0

    sonuc_tablosu_yazdir("Kucuk Ornek (5J-2S-2F)", gbest_set, sure)
    hedef_istatistikleri_yazdir(gbest_set) 
    pareto_cephesi_ciz(
        gbest_set,
        kaydet_yol=os.path.join(CIKTI, 'deney1_pareto.png'))

    yakinlasma_grafigi(
        gecmis,
        kaydet_yol=os.path.join(CIKTI, 'deney1_yakinsama.png'))

    gantt_grafigi(
        gbest_set, inst,
        kaydet_yol=os.path.join(CIKTI, 'deney1_gantt.png'))

    return gbest_set, gecmis

def deney_orta():
    inst = DHHFSPInstance(JN=20, SN=3, FN=3, seed=2025)
    inst.yazdir()

    params = {
        'r1': 0.4, 'r2': 0.4,
        'cv2': 0.2, 'cv3': 0.4, 'cv4': 0.3,
        'mv2': 0.06, 'mv3': 0.10, 'mv4': 0.25,
        'boyutlar': (15, 15, 15, 55),
        'Q_Times': 40, 'L_Times': 40,
        'gamma': 0.8, 'epsilon': 0.9, 'alpha': 0.1
    }

    t0 = time.time()
    gbest_set, gecmis = hmopso_qls(
        inst, N=100, E_max=10000, params=params, verbose=True)
    sure = time.time() - t0

    sonuc_tablosu_yazdir("Orta Olcek (20J-3S-3F)", gbest_set, sure)
    hedef_istatistikleri_yazdir(gbest_set)

    pareto_cephesi_ciz(
        gbest_set,
        kaydet_yol=os.path.join(CIKTI, 'deney2_pareto.png'))

    yakinlasma_grafigi(
        gecmis,
        kaydet_yol=os.path.join(CIKTI, 'deney2_yakinsama.png'))

    gantt_grafigi(
        gbest_set, inst,
        kaydet_yol=os.path.join(CIKTI, 'deney2_gantt.png'))

    return gbest_set, gecmis

def deney_buyuk():
    inst = DHHFSPInstance(JN=50, SN=5, FN=4, seed=2025)
    inst.yazdir()

    params = {
        'r1': 0.4, 'r2': 0.4,
        'cv2': 0.2, 'cv3': 0.4, 'cv4': 0.3,
        'mv2': 0.06, 'mv3': 0.10, 'mv4': 0.25,
        'boyutlar': (15, 15, 15, 55),
        'Q_Times': 40, 'L_Times': 40,
        'gamma': 0.8, 'epsilon': 0.9, 'alpha': 0.1
    }

    t0 = time.time()
    gbest_set, gecmis = hmopso_qls(
        inst, N=100, E_max=20000, params=params, verbose=True)
    sure = time.time() - t0

    sonuc_tablosu_yazdir("Buyuk Olcek (50J-5S-4F)", gbest_set, sure)
    hedef_istatistikleri_yazdir(gbest_set)

    pareto_cephesi_ciz(
        gbest_set,
        kaydet_yol=os.path.join(CIKTI, 'deney3_pareto.png'))

    yakinlasma_grafigi(
        gecmis,
        kaydet_yol=os.path.join(CIKTI, 'deney3_yakinsama.png'))

    return gbest_set, gecmis

def parametrik_karsilastirma():
    inst = DHHFSPInstance(JN=20, SN=3, FN=3, seed=42)

    sonuclar = {}
    for eps in [0.7, 0.9]:
        params = {
            'r1': 0.4, 'r2': 0.4,
            'cv2': 0.2, 'cv3': 0.4, 'cv4': 0.3,
            'mv2': 0.06, 'mv3': 0.10, 'mv4': 0.25,
            'boyutlar': (15, 15, 15, 55),
            'Q_Times': 40, 'L_Times': 40,
            'gamma': 0.8, 'epsilon': eps, 'alpha': 0.1
        }
     
        t0 = time.time()
        gbest_set, _ = hmopso_qls(
            inst, N=100, E_max=8000, params=params, verbose=False)
        sure = time.time() - t0

        hedefler = [c.hedefler for c in gbest_set if c.hedefler]
        sonuclar[eps] = {
            'pareto': len(gbest_set),
            'cmax_min': min(h[0] for h in hedefler),
            'tec_min':  min(h[1] for h in hedefler),
            'twc_min':  min(h[2] for h in hedefler),
            'sure':     sure
        }

    print(f"\n  {'epsilon':<10} {'Pareto':>8} {'C_max_min':>12} "
          f"{'TEC_min':>12} {'TWC_min':>12} {'Sure(s)':>10}")
    for eps, s in sonuclar.items():
        print(f"  {eps:<10} {s['pareto']:>8} {s['cmax_min']:>12.3f} "
              f"{s['tec_min']:>12.3f} {s['twc_min']:>12.3f} "
              f"{s['sure']:>10.2f}")

if __name__ == '__main__':
    gbest1, gecmis1 = deney_kucuk()
    gbest2, gecmis2 = deney_orta()
    gbest3, gecmis3 = deney_buyuk()