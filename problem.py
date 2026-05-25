"""
İsci-Makine-Cevre (WME) Dagitilmis Heterojen Hibrit Akis Atolyesi Problemi (DHHFSP-WME)
Makale: Zhang ve ark. (2026), Journal of Manufacturing Systems 85, 1-21

Tablo 4 ve 5 ornegi ve rastgele uretilen ornekler icin uygulanmistir.

JN  : Is (siparis) sayisi
SN  : Uretim asamasi sayisi
FN  : Fabrika sayisi
MN : Makine sayisi
ST: Standart islem sureleri
MS: Makine isleme hizi
WN: Isci sayisi
WE: Isci verimlilik katsayisi
"""

import numpy as np

class DHHFSPInstance:
    def __init__(self, JN, SN, FN, seed=42):
        self.JN = JN
        self.SN = SN
        self.FN = FN
        np.random.seed(seed)
        self._uret()

    def _uret(self):
        JN, SN, FN = self.JN, self.SN, self.FN

        # Her asama icin her fabrikada 1-3 arasinda random makine sayilari MN[f][k] tanimlanir
        self.MN = [[np.random.randint(1, 4) for _ in range(SN)] for _ in range(FN)]

        # Standart islem sureSİ ST[j][k] = j isinin k asamasindaki standart sureleri
        self.ST = np.random.randint(3, 16, size=(JN, SN)).astype(float)

        # Makine isleme hizi MS[f][k][i] = f fabrikasinin k asamasindaki i makinesinin isleme hizi
        self.MS = []
        for f in range(FN):
            fk = []
            for k in range(SN):
                fk.append([round(np.random.uniform(0.8, 1.5), 2)
                            for _ in range(self.MN[f][k])])
            self.MS.append(fk)

        # Aktif enerji tuketimi MECPU[f][k][i] = f fabrikasinin k asamasindaki i makinesinin aktif enerji tuketimi (kWh/dk)
        self.MECPU = []
        for f in range(FN):
            fk = []
            for k in range(SN):
                fk.append([round(np.random.uniform(5, 12), 2)
                            for _ in range(self.MN[f][k])])
            self.MECPU.append(fk)

        # Bekleme enerji tuketimi MECSU[f][k][i] = f fabrikasinin k asamasindaki i makinesinin beklemeli enerji tuketimi (kWh/dk)
        self.MECSU = []
        for f in range(FN):
            fk = []
            for k in range(SN):
                fk.append([round(self.MECPU[f][k][i] * 0.15, 2)
                            for i in range(self.MN[f][k])])
            self.MECSU.append(fk)

        # Fabrika basi isci sayisi = o fabrikanin toplam makine sayisi
        self.WN = [sum(self.MN[f]) for f in range(FN)]

        # Isci verimlilik katsayisi WE[f][w] = f fabrikasinin w iscisinin verimlilik katsayisi
        self.WE = []
        for f in range(FN):
            self.WE.append([round(np.random.uniform(0.5, 1.5), 2)
                            for _ in range(self.WN[f])])

        # Isci birim zaman maliyeti WCPU[f][w] = WE * 10 olarak tanimlanir
        self.WCPU = []
        for f in range(FN):
            self.WCPU.append([round(self.WE[f][w] * 10, 2)
                               for w in range(self.WN[f])])

    def yazdir(self):
        print(f"DHHFSP-WME Ornegi: {self.JN} is, {self.SN} asama, {self.FN} fabrika")
        print(f"Makine sayilari (fabrika x asama):")
        for f in range(self.FN):
            print(f"Fabrika {f+1}: {self.MN[f]}")
        print(f"Isci sayilari: {self.WN}")
        print(f"ST (ilk 3 is, tum asamalar):")
        for j in range(min(3, self.JN)):
            print(f"    Is {j+1}: {self.ST[j]}")

"""
Makaledeki 4 ve 5 numarali tablonun ornegi icin fonksiyon. DHHFSPInstance nesnesi uretilir ve ilgili parametreler elle atanir.
"""
def kucuk_ornek():
    inst = DHHFSPInstance.__new__(DHHFSPInstance)
    inst.JN = 5
    inst.SN = 2
    inst.FN = 2

    # Tablo 4: Standart islem sureleri
    inst.ST = np.array([
        [8, 4],   # J1
        [3, 5],   # J2
        [7, 4],   # J3
        [6, 3],   # J4
        [4, 3],   # J5
    ], dtype=float)

    # Makine sayilari
    # Fabrika 1: Asama1=2 makine, Asama2=1 makine
    # Fabrika 2: Asama1=1 makine, Asama2=2 makine
    inst.MN = [[2, 1], [1, 2]]

    # Tablo 5: Makine hizlari
    inst.MS = [
        [[1.0, 1.2], [1.2]],  # Fabrika 1: Asama1=2 makine, Asama2=1 makine
        [[1.0], [1.0, 1.2]],  # Fabrika 2: Asama1=1 makine, Asama2=2 makine
    ]

    # Tablo 5: Aktif enerji
    inst.MECPU = [
        [[7, 10], [6]], # Fabrika 1: Asama1=2 makine, Asama2=1 makine
        [[7], [6, 8]],  # Fabrika 2: Asama1=1 makine, Asama2=2 makine
    ]

    # Bekleme enerjisi
    inst.MECSU = [
        [[1.0, 1.5], [0.9]], # Fabrika 1: Asama1=2 makine, Asama2=1 makine
        [[1.0],      [0.9, 1.2]], # Fabrika 2: Asama1=1 makine, Asama2=2 makine
    ]

    # Isci sayisi = toplam makine sayisi
    inst.WN = [3, 3]

    # Tablo 5: Isci verimliligi
    inst.WE = [
        [1.2, 0.8, 1.0],  # Fabrika 1: W1, W2, W3
        [1.1, 0.9, 1.3],  # Fabrika 2: W1, W2, W3
    ]

    # Isci maliyeti
    inst.WCPU = [
        [12, 8, 10],  # Fabrika 1: W1, W2, W3
        [11, 9, 13],  # Fabrika 2: W1, W2, W3
    ]

    return inst
