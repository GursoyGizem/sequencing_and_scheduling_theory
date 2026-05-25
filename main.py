import os
import time
import numpy as np
import matplotlib

from problem      import DHHFSPInstance, kucuk_ornek
from solution     import rastgele_cozum, cozumle
from hmopso_qls   import hmopso_qls
from visualization import (pareto_cephesi_ciz, yakinlasma_grafigi, hedef_istatistikleri_yazdir, gantt_grafigi)

matplotlib.use('Agg')

CIKTI = os.path.join(os.path.dirname(__file__), 'sonuclar')
os.makedirs(CIKTI, exist_ok=True)

params = {
        'r1': 0.4,  'r2': 0.4,
        'cv2': 0.2, 'cv3': 0.4,  'cv4': 0.3,
        'mv2': 0.06,'mv3': 0.10, 'mv4': 0.25,
        'boyutlar': (15, 15, 15, 55),
        'Q_Times': 40, 'L_Times': 40,  # tablo 5 eslesmesi
        'gamma': 0.8, 'epsilon': 0.9, 'alpha': 0.1
}

def sonuc_tablosu_yazdir(etiket, gbest_set, sure):
    hedefler = [c.hedefler for c in gbest_set if c.hedefler]
    if not hedefler:
        print("Sonuc yok")
        return

    cmax_vals = [h[0] for h in hedefler]
    tec_vals  = [h[1] for h in hedefler]
    twc_vals  = [h[2] for h in hedefler]

    print(f" DENEY: {etiket}")
    print(f" Calisma suresi      : {sure:.1f} sn  ({sure/60:.1f} dk)")
    print(f" Pareto cozum sayisi : {len(gbest_set)}")
    print(f" {'Hedef':<8} {'Min':>12} {'Max':>12} {'Ort':>12} {'Std':>12}")

    for isim, vals in [('C_max', cmax_vals), ('TEC', tec_vals), ('TWC', twc_vals)]:
        print(f"  {isim:<8} {min(vals):>12.4f} {max(vals):>12.4f} "
              f"{np.mean(vals):>12.4f} {np.std(vals):>12.4f}")

def deney_kucuk():
    inst = kucuk_ornek()
    inst.yazdir()

    # N=30 icin boyutlar otomatik (4,4,4,18)'e ayarlanir — bunu acikca belirt
    kucuk_params = {**params, 'boyutlar': (4, 4, 4, 18)}

    t0 = time.time()
    gbest_set, gecmis = hmopso_qls(
        inst, N=30, E_max=5000, params=kucuk_params, verbose=True)
    sure = time.time() - t0

    sonuc_tablosu_yazdir("Kucuk Ornek — 5J-2S-2F  [Makale Tablo 4&5]",
                          gbest_set, sure)
    hedef_istatistikleri_yazdir(gbest_set)

    pareto_cephesi_ciz(gbest_set,
        kaydet_yol=os.path.join(CIKTI, 'deney1_pareto.png'))
    yakinlasma_grafigi(gecmis,
        kaydet_yol=os.path.join(CIKTI, 'deney1_yakinsama.png'))
    gantt_grafigi(gbest_set, inst,
        kaydet_yol=os.path.join(CIKTI, 'deney1_gantt.png'))

    return gbest_set, gecmis

def deney_orta():
    inst = DHHFSPInstance(JN=20, SN=3, FN=3, seed=2025)
    inst.yazdir()

    t0 = time.time()
    gbest_set, gecmis = hmopso_qls(
        inst, N=100, E_max=5000, params=params, verbose=True)
    sure = time.time() - t0

    sonuc_tablosu_yazdir("Orta Olcek — 20J-3S-3F",
                          gbest_set, sure)
    hedef_istatistikleri_yazdir(gbest_set)

    pareto_cephesi_ciz(gbest_set,
        kaydet_yol=os.path.join(CIKTI, 'deney2_pareto.png'))
    yakinlasma_grafigi(gecmis,
        kaydet_yol=os.path.join(CIKTI, 'deney2_yakinsama.png'))
    gantt_grafigi(gbest_set, inst,
        kaydet_yol=os.path.join(CIKTI, 'deney2_gantt.png'))

    return gbest_set, gecmis

if __name__ == '__main__':
    gbest1, gecmis1 = deney_kucuk()
    gbest2, gecmis2 = deney_orta()