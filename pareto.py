import numpy as np

"""
Tum hedeflerde a <= b VE en az bir hedefte a < b ise EVET.
"""
def baskin_mi(a, b):
    return (all(x <= y for x, y in zip(a, b)) and any(x <  y for x, y in zip(a, b)))

"""
Cozum listesinden baskin olmayan (Pareto-optimal) cozumleri dondur.
"""
def baskin_olmayanlar(cozumler):
    n = len(cozumler)
    baskin_olmayan = []
    for i, ci in enumerate(cozumler):
        if ci.hedefler is None:
            continue
        iyisi_var_mi = False
        for j, cj in enumerate(cozumler):
            if i == j or cj.hedefler is None:
                continue
            if baskin_mi(cj.hedefler, ci.hedefler):
                iyisi_var_mi = True
                break
        if not iyisi_var_mi:
            baskin_olmayan.append(ci)
    return baskin_olmayan

"""
PDDR-FF (Baskinlik Degerlendirme Fonksiyonu):
eval(X) = |q(X)| + |p(X)| + 1
|p(X)| = X'nin baskin oldugu cozum sayisi
|q(X)| = X'yi baskin eden cozum sayisi
Kucuk deger -> Daha guclu cozum -> G4'e secilir
"""
def pddr_ff(cozum, arsiv):
    if cozum.hedefler is None:
        return float('inf')

    p = sum(1 for c in arsiv
            if c.hedefler and baskin_mi(cozum.hedefler, c.hedefler))
    q = sum(1 for c in arsiv
            if c.hedefler and baskin_mi(c.hedefler, cozum.hedefler))
    return q + p + 1  

"""
Pareto arsivini guncelle:
    1. Yeni cozumleri ekle
    2. Baskin olmayanlari sec
    3. Boyut sinirini kor
"""
def arsiv_guncelle(arsiv, yeni_cozumler, maks_boyut=200):
    aday = arsiv + [c for c in yeni_cozumler if c.hedefler is not None]
    pareto = baskin_olmayanlar(aday)

    # Boyut siniri: cok kalabalasirsa kirpintiyi sec
    if len(pareto) > maks_boyut:
        # Dagitim skoru hesapla - kalabalik bolgeleri kirp
        pareto = kirp(pareto, maks_boyut)

    return pareto

"""
Pareto arsivini hedef boyuta kirp.
En kalabalik komsuluktaki cozumleri kaldir.
"""
def kirp(cozumler, hedef_boyut):
    while len(cozumler) > hedef_boyut:
        # Her cozum icin en yakin komsusuna mesafe hesapla
        hedefler = np.array([c.hedefler for c in cozumler])
        # Normalize et
        mn = hedefler.min(0) + 1e-10
        mx = hedefler.max(0) + 1e-10
        norm = (hedefler - mn) / (mx - mn)

        # Her noktanin en yakin komsusuna mesafe
        min_mesafe = []
        for i in range(len(norm)):
            mesafeler = [np.linalg.norm(norm[i] - norm[j])
                         for j in range(len(norm)) if i != j]
            min_mesafe.append(min(mesafeler) if mesafeler else 0)

        # En kucuk mesafeli (en kalabalik komsu) noktayi kaldir
        kaldir = int(np.argmin(min_mesafe))
        cozumler.pop(kaldir)

    return cozumler

"""Arsivdeki hedefleri [0,1] araligina normalize et."""
def normalize_hedefler(arsiv):
    hedefler = np.array([c.hedefler for c in arsiv if c.hedefler])
    if len(hedefler) == 0:
        return hedefler
    mn = hedefler.min(0)
    mx = hedefler.max(0)
    aralik = mx - mn
    aralik[aralik == 0] = 1
    return (hedefler - mn) / aralik