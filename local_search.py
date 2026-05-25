"""
Fabrikalar Arasi ve Fabrika Ici Yerel Arama
- VNS 9 operatoru (JS, MA, WA x Ekleme, Ters Cevirme, Takas)
    VNS operatorunu uygula (1-9 arasi).
    op_no: 0=JS-Ekleme, 1=JS-Ters, 2=JS-Takas,
           3=MA-Ekleme, 4=MA-Ters, 5=MA-Takas,
           6=WA-Ekleme, 7=WA-Ters, 8=WA-Takas
    fabrika_idx: sadece bu fabrikadaki isler uzerinde islem yap

- Q-ogrenme tabanli fabrika secimi
"""

import numpy as np
import random
from solution import Cozum, rastgele_cozum, cozumle
from pareto import baskin_mi, pddr_ff

"""pos2'deki elemani al, pos1'in onune ekle"""
def ekleme(dizi, pos1, pos2):
    d = dizi.copy()
    eleman = d.pop(pos2)
    d.insert(pos1, eleman)
    return d

"""pos1 ile pos2 arasindaki alt diziyi ters cevir."""
def ters_cevir(dizi, pos1, pos2):
    d = dizi.copy()
    p1, p2 = min(pos1, pos2), max(pos1, pos2)
    d[p1:p2+1] = d[p1:p2+1][::-1]
    return d

"""pos1 ve pos2'deki elemanlari yer degistir."""
def takas(dizi, pos1, pos2):
    d = dizi.copy()
    d[pos1], d[pos2] = d[pos2], d[pos1]
    return d

OPERATORLER = [ekleme, ters_cevir, takas]
OPERATOR_ADLARI = ['Ekleme', 'TersCevirme', 'Takas']

def vns_operatoru_uygula(cozum, op_no, inst, fabrika_idx=None):
    yeni = cozum.kopyala()
    vektor_no = op_no // 3   # 0=JS, 1=MA, 2=WA
    op_tipi = op_no % 3      # 0=Ekleme, 1=TersCevirme, 2=Takas
    op = OPERATORLER[op_tipi]

    if vektor_no == 0:  # JS vektoru
        if fabrika_idx is not None:
            # Sadece secili fabrikadaki islerin indekslerini al
            idx_listesi = [i for i, j in enumerate(yeni.JS)
                           if yeni.FA[j] == fabrika_idx]
        else:
            idx_listesi = list(range(len(yeni.JS)))

        if len(idx_listesi) < 2:
            return yeni
        pos1, pos2 = sorted(random.sample(idx_listesi, 2))
        # JS uzerinde isleme
        alt = [yeni.JS[i] for i in idx_listesi]
        p1l = idx_listesi.index(pos1)
        p2l = idx_listesi.index(pos2)
        alt_yeni = op(alt, p1l, p2l)
        for i, idx in enumerate(idx_listesi):
            yeni.JS[idx] = alt_yeni[i]

    elif vektor_no == 1:  # MA vektoru
        if fabrika_idx is not None:
            isler = [j for j in range(inst.JN)
                     if yeni.FA[j] == fabrika_idx]
        else:
            isler = list(range(inst.JN))

        if len(isler) < 2:
            return yeni
        j1, j2 = random.sample(isler, 2)
        # Rastgele asama sec
        k = random.randint(0, inst.SN - 1)
        f = yeni.FA[j1]
        m1 = yeni.MA[j1][k] % inst.MN[f][k]
        m2 = yeni.MA[j2][k] % inst.MN[yeni.FA[j2]][k]

        if op_tipi == 0:  # Ekleme: j2'nin k.asamasini j1'e koy
            yeni.MA[j1][k] = m2
        elif op_tipi == 1:  # Ters cevirme: isler arasi ters cevir
            yeni.MA[j1][k], yeni.MA[j2][k] = m2, m1
        else:  # Takas
            yeni.MA[j1][k], yeni.MA[j2][k] = m2, m1

    else:  # WA vektoru
        if fabrika_idx is not None:
            f = fabrika_idx
        else:
            f = random.randint(0, inst.FN - 1)

        wa = yeni.WA[f]
        if len(wa) < 2:
            return yeni
        p1, p2 = sorted(random.sample(range(len(wa)), 2))
        yeni.WA[f] = op(wa, p1, p2)

    return yeni

"""
Fabrikalar Arasi Yerel Arama (Algoritma 4)
alt_suru_no: 0=G1(Cmax), 1=G2(TEC), 2=G3(TWC), 3=G4(denge)
- Kritik fabrikayi bul: en yuksek (F_max) ve en dusuk (F_min) yuklu fabrikalari bul.
  hedef_idx: 0=Cmax, 1=TEC, 2=TWC
- Takas: F_max'tan bir is al, F_min'e ver (takas).
- Ekleme: F_max'tan bir isi F_min'e tasit.  
"""

def kritik_fabrika_bul(cozum, inst, hedef_idx):
    fabrika_yuk = {f: 0.0 for f in range(inst.FN)}
    for j in range(inst.JN):
        fabrika_yuk[cozum.FA[j]] += inst.ST[j].sum()

    sirali = sorted(fabrika_yuk.items(), key=lambda x: x[1])
    F_min = sirali[0][0]
    F_max = sirali[-1][0]

    if F_min == F_max:
        F_min = (F_max + 1) % inst.FN

    return F_max, F_min


def kritik_fabrika_takas(cozum, inst, hedef_idx):
    F_max, F_min = kritik_fabrika_bul(cozum, inst, hedef_idx)
    isler_max = [j for j in range(inst.JN) if cozum.FA[j] == F_max]
    isler_min = [j for j in range(inst.JN) if cozum.FA[j] == F_min]

    if not isler_max or not isler_min:
        return cozum.kopyala()

    yeni = cozum.kopyala()
    j1 = random.choice(isler_max)
    j2 = random.choice(isler_min)
    yeni.FA[j1] = F_min
    yeni.FA[j2] = F_max
    # Makine atamasini guncelle (fabrika degisince makine gecersiz olabilir)
    for k in range(inst.SN):
        yeni.MA[j1][k] = yeni.MA[j1][k] % inst.MN[F_min][k]
        yeni.MA[j2][k] = yeni.MA[j2][k] % inst.MN[F_max][k]
    return yeni


def kritik_fabrika_ekle(cozum, inst, hedef_idx):
    F_max, F_min = kritik_fabrika_bul(cozum, inst, hedef_idx)
    isler_max = [j for j in range(inst.JN) if cozum.FA[j] == F_max]

    if not isler_max:
        return cozum.kopyala()

    yeni = cozum.kopyala()
    j1 = random.choice(isler_max)
    yeni.FA[j1] = F_min
    for k in range(inst.SN):
        yeni.MA[j1][k] = yeni.MA[j1][k] % inst.MN[F_min][k]
    return yeni


def fabrikalar_arasi_yerel_arama(cozum, inst, alt_suru_no, arsiv):
    hedef_idx = min(alt_suru_no, 2)  # G4 icin 0 kullan
    en_iyi = cozum.kopyala()

    for t in range(1, 3):  # 2 operator: takas ve ekleme
        if t == 1:
            aday = kritik_fabrika_takas(en_iyi, inst, hedef_idx)
        else:
            aday = kritik_fabrika_ekle(en_iyi, inst, hedef_idx)

        cozumle(aday, inst)

        kabul = False
        if alt_suru_no < 3:  # G1, G2, G3
            if (en_iyi.hedefler and aday.hedefler and
                    aday.hedefler[hedef_idx] < en_iyi.hedefler[hedef_idx]):
                en_iyi = aday
                kabul = True
        else:  # G4
            aday_eval = pddr_ff(aday, arsiv)
            en_iyi_eval = pddr_ff(en_iyi, arsiv)
            if aday_eval <= en_iyi_eval:
                en_iyi = aday
                kabul = True

        if not kabul:
            continue  # bir sonraki operator

    return en_iyi

"""
Q-Ogrenme tabanli Fabrika Ici Yerel Arama (Algoritma 5-6)

Q-ogrenme ajani: hangi fabrikada VNS uygulanacagini ogrenir
Durum sayisi: 8 (her alt suru icin 2)
Eylem sayisi: FN (fabrika sayisi)

"""
class QOgrenmeAjani:
    def __init__(self, FN, alt_suru_no, alpha=0.1, gamma=0.8, epsilon=0.9):
        self.FN = FN
        self.alt_suru_no = alt_suru_no
        self.alpha = alpha      # Ogrenme hizi
        self.gamma = gamma      # Indirim faktoru
        self.epsilon = epsilon  # Acgozluluk orani

        # Q-tablosu: 8 durum x FN eylem
        self.Q = np.zeros((8, FN))
        self.durum = 1  # Baslangic durumu

    """
    Delta_f'e gore durumu belirle.
    delta_f > 0: iyilesme var -> durum 0
    delta_f <= 0: iyilesme yok -> durum 1
    Alt suru offseti: G1=0, G2=2, G3=4, G4=6
    """
    def durum_belirle(self, delta_f):
        offset = min(self.alt_suru_no, 3) * 2
        if delta_f > 0:
            return offset + 0
        else:
            return offset + 1

    def eylem_sec(self):
        if random.random() < self.epsilon:
            # Somurme: Q-tablosundan en iyi eylemi sec
            return int(np.argmax(self.Q[self.durum]))
        else:
            # Kesif: rastgele fabrika sec
            return random.randint(0, self.FN - 1)

    """Bellman denklemi ile Q-tablosunu guncelle."""
    def q_tablosu_guncelle(self, durum, eylem, odul, yeni_durum):
        mevcut_q = self.Q[durum, eylem]
        en_iyi_gelecek = np.max(self.Q[yeni_durum])
        yeni_q = mevcut_q + self.alpha * (
            odul + self.gamma * en_iyi_gelecek - mevcut_q
        )
        self.Q[durum, eylem] = yeni_q


def odul_hesapla(onceki, sonraki, alt_suru_no):
    if onceki is None or sonraki is None:
        return 0.0

    if alt_suru_no == 0:   # G1: Cmax iyilesmesi
        return onceki[0] - sonraki[0]
    elif alt_suru_no == 1: # G2: TEC iyilesmesi
        return onceki[1] - sonraki[1]
    elif alt_suru_no == 2: # G3: TWC iyilesmesi
        return onceki[2] - sonraki[2]
    else:                  # G4: Toplam goreceli iyilesme
        toplam = 0.0
        for i in range(3):
            if onceki[i] > 0:
                toplam += (onceki[i] - sonraki[i]) / onceki[i]
        return toplam

"""
Algoritma 6: VNS Fabrika Ici Arama
"""
def vns_fabrika_ici(cozum, inst, alt_suru_no, arsiv, fabrika_idx, L_Times=40):
    en_iyi = cozum.kopyala()
    s = 0
    s_max = 9  

    iterasyon = 0
    while s < s_max and iterasyon < L_Times:
        iterasyon += 1

        # Sallama: s. komsuluK operatorunu uygula
        aday = vns_operatoru_uygula(en_iyi, s, inst, fabrika_idx)
        cozumle(aday, inst)

        # Kabul kriteri
        kabul = False
        if alt_suru_no < 3:  # G1, G2, G3: tek hedef
            hedef_idx = alt_suru_no
            if (en_iyi.hedefler and aday.hedefler and
                    aday.hedefler[hedef_idx] < en_iyi.hedefler[hedef_idx]):
                en_iyi = aday
                kabul = True
        else:  # G4: PDDR-FF
            aday_eval = pddr_ff(aday, arsiv)
            en_iyi_eval = pddr_ff(en_iyi, arsiv)
            if aday_eval <= en_iyi_eval:
                en_iyi = aday
                kabul = True

        if kabul:
            s = 0  # Basarili: sifirdan basla
        else:
            s += 1  # Basarisiz: bir sonraki komsuluK

    return en_iyi

"""
Algoritma 5: Q-ogrenme tabanli fabrika ici arama.
Ajan hangi fabrikada VNS yapacagini ogreniri.
"""
def q_ogrenme_fabrika_ici_arama(cozum, inst, alt_suru_no, arsiv, Q_Times=40, L_Times=40, alpha=0.1, gamma=0.8, epsilon=0.9):
    ajan = QOgrenmeAjani(inst.FN, alt_suru_no, alpha, gamma, epsilon)
    en_iyi = cozum.kopyala()

    for i in range(Q_Times):
        durum = ajan.durum
        eylem = ajan.eylem_sec()  # fabrika secilir

        # Secilen fabrikada VNS uygula
        fabrika_idx = eylem % inst.FN
        aday = vns_fabrika_ici(en_iyi, inst, alt_suru_no, arsiv, fabrika_idx, L_Times)

        # Odul hesapla
        odul = odul_hesapla(en_iyi.hedefler, aday.hedefler, alt_suru_no)

        # Yeni durumu belirle
        delta_f = odul
        yeni_durum = ajan.durum_belirle(delta_f)

        # Q-tablosunu guncelle
        ajan.q_tablosu_guncelle(durum, eylem, odul, yeni_durum)
        ajan.durum = yeni_durum

        # Cozumu guncelle
        if odul > 0:
            en_iyi = aday

    return en_iyi
