# DHHFSP-WME | HMOPSO-QLS Python Implementasyonu

> **Zhang, Y., ve ark. (2026).** *A hybrid multiobjective particle swarm optimization with Q-learning-driven local search for distributed heterogeneous hybrid flow-shop scheduling problem with worker–machine–environment collaboration.* Journal of Manufacturing Systems, 85, 1–21.

---
## Problem Tanımı

**DHHFSP-WME** (Distributed Heterogeneous Hybrid Flow-Shop Scheduling Problem with
Worker–Machine–Environment Collaboration), gerçek üretim ortamlarının karmaşıklığını
yansıtan üç boyutlu bir çizelgeleme problemidir.

### Temel Özellikler

| Boyut | Açıklama |
|-------|----------|
| **Dağıtık fabrikalar** | İşler birden fazla fabrikaya atanabilir; fabrikalar heterojendir |
| **Hibrit akış atölyesi** | Her fabrikada birden fazla paralel makine içeren ardışık aşamalar bulunur |
| **İşçi–makine işbirliği** | Her makine bir işçi tarafından kullanılır; işçi verimliliği işlem süresini etkiler |
| **Çevre faktörü** | Enerji tüketimi makine hızı, bekleme durumu ve işçi verimliliğine bağlı hesaplanır |

### Hedef

```
minimize  C_max  =  toplam tamamlanma süresi (makespan)
minimize  TEC    =  toplam enerji tüketimi
minimize  TWC    =  toplam işçi maliyeti
```

Bu üç hedef arasında doğrusal olmayan bir çatışma mevcuttur:
- Hızlı makineler → düşük C_max, yüksek TEC
- Verimli işçiler → düşük C_max, yüksek TWC
- Düşük enerjili çalışma → yüksek C_max

---

## Algoritma

**HMOPSO-QLS** (Hybrid Multiobjective Particle Swarm Optimization with Q-Learning-driven Local Search), PSO tabanlı global arama ile Q-öğrenme destekli yerel arama mekanizmalarını birleştiren hibrit bir meta-sezgisel algoritmadır.

### Çözüm Kodlaması

Her parçacık dört vektörle temsil edilir:

```
JS  →  İş sırası (permütasyon)            örn. [2, 0, 4, 1, 3]
FA  →  Fabrika ataması (her iş için)       örn. [0, 1, 0, 0, 1]
MA  →  Makine ataması (her iş × aşama)    örn. [[1,0], [0,1], ...]
WA  →  İşçi ataması (her fabrika için)    örn. [[2,0,1], [1,2,0]]
```

### Algoritma Akışı

```
1. Başlangıç popülasyonu üret (N parçacık)
   └─ Rastgele JS, FA, MA, WA vektörleri
   └─ Sağ-öteleme stratejisi ile TEC/TWC iyileştir

2. Her iterasyonda:
   ├─ Sürü Ayrıştırması (4 alt-sürü)
   │   ├─ G1 (15): C_max odaklı sınır sürüsü
   │   ├─ G2 (15): TEC odaklı sınır sürüsü
   │   ├─ G3 (15): TWC odaklı sınır sürüsü
   │   └─ G4 (55): PDDR-FF ile denge sürüsü
   │
   ├─ PSO Güncellemesi (Küresel Arama)
   │   ├─ JS: Değişim dizisi operatörü
   │   ├─ FA/MA/WA: İki noktalı çaprazlama + mutasyon
   │   └─ Her alt-sürü kendi gbest'ine yönelir
   │
   ├─ Fabrikalar Arası Yerel Arama
   │   ├─ Kritik fabrika tespiti (en yüklü / en az yüklü)
   │   ├─ Kritik Fabrika Takas operatörü
   │   └─ Kritik Fabrika Ekleme operatörü
   │
   ├─ Q-Öğrenme Tabanlı Fabrika İçi Arama
   │   ├─ Q-tablosu: 8 durum × FN eylem
   │   ├─ ε-greedy politika (keşif/sömürü dengesi)
   │   ├─ Bellman denklemi ile Q-tablo güncelleme
   │   └─ VNS: 9 komşuluk yapısı (JS/MA/WA × Ekleme/TersCevirme/Takas)
   │
   └─ Pareto Arşiv Yönetimi
       ├─ PDDR-FF ile yoğunluk-mesafe değerlendirmesi
       └─ Maksimum 200 baskın olmayan çözüm

3. Yakınsama → Pareto cephesi döndür
```

---

## Gereksinimler

```
Python >= 3.8
numpy
matplotlib
```

Kurulum:

```bash
pip install numpy matplotlib
```

---

## Kurulum ve Çalıştırma

### 1. Depoyu klonla

```bash
git clone https://github.com/kullanici_adi/dhhfsp-wme-hmopso-qls.git
cd dhhfsp-wme-hmopso-qls
```

### 2. Çalıştır

```bash
python main.py
```
---

## Deney Sonuçları

### Deney 1 — Küçük Örnek (5 İş, 2 Aşama, 2 Fabrika)

Makale Tablo 4 & 5 ile birebir aynı problem verisi kullanılmıştır.

| Parametre | Değer |
|-----------|-------|
| N (parçacık) | 30 |
| E_max | 5.000 |
| Q_Times / L_Times | 40 / 40 |

| Hedef | Min | Max | Ortalama | 
|-------|-----|-----|----------|
| C_max | 12.8611 | 26.389 | 17.440 |
| TEC | 255.728 | 333.808 | 289.231 | 
| TWC | 391.667 | 463.333 | 433.775 | 

Çalışma süresi: 16.6 dk
Pareto Çözüm Sayısı: 200

### Deney 2 — Orta Ölçek (20 İş, 3 Aşama, 3 Fabrika)

| Parametre | Değer |
|-----------|-------|
| N (parçacık) | 100 |
| E_max | 5.000 |
| Q_Times / L_Times | 40 / 40 |
| boyutlar (G1/G2/G3/G4) | 15/15/15/55 |

| Hedef | Min | Max | Ortalama | 
|-------|-----|-----|----------|
| C_max | 65.036 | 153.351 | 95.203 |
| TEC | 3723.2840 | 5504.03 | 4727.035 | 
| TWC | 4333.345 | 5114.190 | 4681.417 | 

Çalışma süresi: 40.8 dk
Pareto Çözüm Sayısı: 200

---

## Parametreler

| Parametre | Sembol | Değer | Açıklama |
|-----------|--------|-------|----------|
| Parçacık sayısı | N | 100 | Toplam sürü büyüklüğü |
| Sınır alt-sürüsü | G1/G2/G3 | 15 | C_max / TEC / TWC odaklı |
| Denge alt-sürüsü | G4 | 55 | PDDR-FF ile merkez sürü |
| pbest yönelim | r1 | 0.4 | Kişisel en iyi çekim oranı |
| gbest yönelim | r2 | 0.4 | Küresel en iyi çekim oranı |
| FA çaprazlama | cv2 | 0.2 | Fabrika atama çaprazlama oranı |
| MA çaprazlama | cv3 | 0.4 | Makine atama çaprazlama oranı |
| WA çaprazlama | cv4 | 0.3 | İşçi atama çaprazlama oranı |
| Q-öğrenme adım | Q_Times | 40 | Q-ajan iterasyon sayısı |
| VNS döngüsü | L_Times | 40 | Yerel arama tekrar sayısı |
| İskonto faktörü | γ | 0.8 | Q-öğrenme gelecek ağırlığı |
| Keşif olasılığı | ε | 0.9 | ε-greedy politika parametresi |
| Öğrenme hızı | α | 0.1 | Q-tablo güncelleme katsayısı |
| Pareto arşiv | — | 200 | Maksimum Pareto çözüm sayısı |

---

## Referans

```bibtex
@article{zhang2026hmopso,
  title   = {A hybrid multiobjective particle swarm optimization with
             Q-learning-driven local search for distributed heterogeneous
             hybrid flow-shop scheduling problem with worker--machine--environment
             collaboration},
  author  = {Zhang, Y. and others},
  journal = {Journal of Manufacturing Systems},
  volume  = {85},
  pages   = {1--21},
  year    = {2026}
}
```

---
*Bu uygulama, Zhang ve ark. (2026) makalesinin implementasyonudur. Bu implementasyon akademik eğitim ve araştırma amacıyla geliştirilmiştir. Orijinal araştırmacılarla veya makalenin yayımlandığı dergi ile doğrudan bir bağlantısı bulunmamaktadır.