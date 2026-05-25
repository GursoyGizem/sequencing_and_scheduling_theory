import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import warnings
from mpl_toolkits.mplot3d import Axes3D
from solution import cozumle

matplotlib.use('Agg') 
warnings.filterwarnings('ignore', category=UserWarning)

NAVY   = '#1A3A5C'
GREEN  = '#145A32'
RED    = '#7B241C'
PURPLE = '#5B2C6F'
GOLD   = '#B7770D'
TEAL   = '#0E6655'
GRAY   = '#2C3E50'
LIGHT  = '#F8F9FA'

def pareto_cephesi_ciz(gbest_set, kaydet_yol=None):
    if not gbest_set:
        print("Pareto cephesi yok")
        return

    hedefler = np.array([c.hedefler for c in gbest_set])
    cmax = hedefler[:, 0]
    tec  = hedefler[:, 1]
    twc  = hedefler[:, 2]

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(LIGHT)
    fig.suptitle('DHHFSP-WME — HMOPSO-QLS Pareto Cephesi Sonuclari', fontsize=14, fontweight='bold', color=NAVY, y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax3d = fig.add_subplot(gs[0, :], projection='3d')
    sc = ax3d.scatter(cmax, tec, twc, c=twc, cmap='plasma', s=80, alpha=0.85, edgecolors='white', linewidths=0.5)
    ax3d.set_xlabel('C_max (Makespan)', fontsize=10, color=GRAY)
    ax3d.set_ylabel('TEC (Enerji)', fontsize=10, color=GRAY)
    ax3d.set_zlabel('TWC (Isci Maliyeti)', fontsize=10, color=GRAY)
    ax3d.set_title(f'3 Boyutlu Pareto Cephesi  ({len(gbest_set)} cozum)', fontsize=11, color=NAVY, fontweight='bold')
    plt.colorbar(sc, ax=ax3d, label='TWC degeri', shrink=0.5)

    proj_params = [
        (gs[1, 0], cmax, tec,  'C_max (Makespan)', 'TEC (Enerji)',        NAVY,   'C_max vs TEC'),
        (gs[1, 1], cmax, twc,  'C_max (Makespan)', 'TWC (Isci Maliyeti)', PURPLE, 'C_max vs TWC'),
        (gs[1, 2], tec,  twc,  'TEC (Enerji)',     'TWC (Isci Maliyeti)', TEAL,   'TEC vs TWC'),
    ]

    for spec, x, y, xlabel, ylabel, color, title in proj_params:
        ax = fig.add_subplot(spec)
        ax.set_facecolor('white')
        ax.scatter(x, y, c=color, s=60, alpha=0.8, edgecolors='white', linewidths=0.5)

        sirali = sorted(zip(x, y))
        px, py = zip(*sirali)
        ax.plot(px, py, color=color, lw=1.2, alpha=0.5, linestyle='--')

        ax.set_xlabel(xlabel, fontsize=9, color=GRAY)
        ax.set_ylabel(ylabel, fontsize=9, color=GRAY)
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if kaydet_yol:
        plt.savefig(kaydet_yol, dpi=180, bbox_inches='tight',facecolor=LIGHT)
        print(f"  Gorsel kaydedildi: {kaydet_yol}")
    plt.show()

def yakinlasma_grafigi(gecmis, kaydet_yol=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(LIGHT)
    fig.suptitle('HMOPSO-QLS — Yakinsama Grafikleri',fontsize=13, fontweight='bold', color=NAVY)

    iterasyonlar = range(1, len(gecmis['cmax']) + 1)

    grafik_params = [
        (axes[0, 0], gecmis['cmax'],        'C_max (En Kucuk)',         NAVY,   'C_max'),
        (axes[0, 1], gecmis['tec'],          'TEC (En Kucuk)',           TEAL,   'TEC'),
        (axes[1, 0], gecmis['twc'],          'TWC (En Kucuk)',           PURPLE, 'TWC'),
        (axes[1, 1], gecmis['pareto_boyut'], 'Pareto Cephesi Boyutu',    GREEN,  'Boyut'),
    ]

    for ax, veri, baslik, renk, etiket in grafik_params:
        ax.set_facecolor('white')
        ax.plot(iterasyonlar, veri, color=renk, lw=2.0, label=etiket)
        ax.fill_between(iterasyonlar, veri, alpha=0.15, color=renk)
        ax.set_title(baslik, fontsize=11, color=renk, fontweight='bold')
        ax.set_xlabel('Iterasyon', fontsize=9, color=GRAY)
        ax.set_ylabel(etiket, fontsize=9, color=GRAY)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if kaydet_yol:
        plt.savefig(kaydet_yol, dpi=180, bbox_inches='tight', facecolor=LIGHT)
        print(f"  Gorsel kaydedildi: {kaydet_yol}")
    plt.show()

def hedef_istatistikleri_yazdir(gbest_set):
    if not gbest_set:
        print("Pareto cephesi bos!")
        return

    hedefler = np.array([c.hedefler for c in gbest_set])
    print(f"Pareto cephesi istatistikleri:")
    print(f"Cozum sayisi: {len(gbest_set)}")
    print(f"  {'Hedef':<12} {'Min':>12} {'Max':>12} {'Ort':>12}")
    isimler = ['C_max', 'TEC', 'TWC']
    for i, isim in enumerate(isimler):
        print(f"  {isim:<12} {hedefler[:,i].min():>12.3f} "
              f"{hedefler[:,i].max():>12.3f} "
              f"{hedefler[:,i].mean():>12.3f}")

def gantt_grafigi(gbest_set, inst, kaydet_yol=None):
    if not gbest_set:
        return
    
    en_iyi = min(gbest_set, key=lambda c: c.hedefler[0])
    JN, SN, FN = inst.JN, inst.SN, inst.FN
    JS, FA, MA, WA = en_iyi.JS, en_iyi.FA, en_iyi.MA, en_iyi.WA

    fabrika_isler = [[] for _ in range(FN)]
    for j in JS:
        fabrika_isler[FA[j]].append(j)

    renkler = plt.cm.Set3(np.linspace(0, 1, JN))

    fig, axes = plt.subplots(FN, 1, figsize=(14, 3 * FN))
    if FN == 1:
        axes = [axes]
    fig.patch.set_facecolor(LIGHT)
    fig.suptitle(f'Gantt Grafigi (En Iyi C_max = {en_iyi.hedefler[0]:.2f})', fontsize=12, fontweight='bold', color=NAVY)

    for f in range(FN):
        ax = axes[f]
        ax.set_facecolor('white')

        mak_zaman = [[0.0] * inst.MN[f][k] for k in range(SN)]
        is_bitis = {j: 0.0 for j in fabrika_isler[f]}

        for k in range(SN):
            if k == 0:
                siralama = fabrika_isler[f]
            else:
                siralama = sorted(fabrika_isler[f],
                                  key=lambda j: is_bitis.get(j, 0))

            for j in siralama:
                m = MA[j][k] % inst.MN[f][k]
                w_idx = (sum(inst.MN[f][kk] for kk in range(k)) + m) % inst.WN[f]
                w = WA[f][w_idx]
                at = inst.ST[j][k] / (inst.MS[f][k][m] * inst.WE[f][w])

                if k == 0:
                    bas = mak_zaman[k][m]
                else:
                    bas = max(is_bitis.get(j, 0), mak_zaman[k][m])

                bit = bas + at
                is_bitis[j] = bit
                mak_zaman[k][m] = bit

                y = k * max(inst.MN[f]) + m
                ax.barh(y, at, left=bas,
                        color=renkler[j], edgecolor='white',
                        linewidth=0.8, alpha=0.85)
                ax.text(bas + at/2, y, f'J{j+1}',
                        ha='center', va='center',
                        fontsize=7, color='black', fontweight='bold')

        yticks = []
        ylabels = []
        for k in range(SN):
            for m in range(inst.MN[f][k]):
                yticks.append(k * max(inst.MN[f]) + m)
                ylabels.append(f'A{k+1}-M{m+1}')

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xlabel('Zaman', fontsize=9, color=GRAY)
        ax.set_title(f'Fabrika {f+1}', fontsize=10,
                     color=NAVY, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    if kaydet_yol:
        plt.savefig(kaydet_yol, dpi=180, bbox_inches='tight',
                    facecolor=LIGHT)
        print(f"  Gorsel kaydedildi: {kaydet_yol}")
    plt.show()
