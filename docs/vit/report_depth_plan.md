# ViT Raporu — Derinlik & Literatür-Temellendirme Planı (Slice A–E sonrası)

Bu dosya, raporun **içerik derinliği**, **alt-bölüm granülerliği** ve
**literatür-temellendirmesi** eksenindeki bölüm-bazlı denetiminden (2026-06-25)
çıkan bulguları dilimlere gruplar. Slice A–E (stil, zorunlu tablolar, atıf
stub'ları, füzyon denklemleri, appendix, kapak) zaten uygulandı; bu plan onları
**tekrar etmez** — yalnızca derinlik/granülerlik/literatür boşluklarını ele alır.

**Kapsam kuralları (tüm dilimler):** inference-only, 0 A100; **sayı/sonuç/atıf
uydurulmaz**; şampiyon `11_triple_weighted` ve kilitli kararlar değişmez; akademik
Türkçe ("seed"); gövdede repo-path/VLD/self-referential ifade yok; literatür
bağlamsal, SOTA yok; her yeni atıf **mevcut non-empty stub'a** eşlenmeli (yoksa
işaretle). Her dilim sonrası `cd reports/vit && tectonic main.tex` temiz derleme.
Onay-kapılı; otomatik ilerleme yok.

## Denetim ölçümleri (özet)

| Bölüm | Kelime | Alt bölüm | Not |
|---|---:|---:|---|
| Giriş | 272 | 0 | düz; backbone seçimi gerekçesi sığ |
| İlgili Çalışmalar | 248 | 4 ¶ | omurga/füzyon paragrafları sığ; imbalance literatürü yok |
| Yöntem | 516 | 5 | "Omurgalar" alt bölümü ~80 kelime (çok kısa); transfer gerekçesi eksik |
| Deneyler | 414 | 5 | sampler/bootstrap mekanizması tanımsız |
| Sonuçlar | 1176 | 5 | güçlü, tablo-yoğun |
| **Tartışma** | **583** | 5 | **en ağırlıklı bölüm ama ince**; hata analizi + imbalance alt bölümü yok |
| Sonuç | 222 | 2 ¶ | kısa; gelecek-iş atıfsız |
| Ekler | 383 | 2 | yeterli |

**Kullanılabilir ama az/hiç kullanılmayan stub'lar:** omurga makaleleri
(ViT/Swin/BEiT — sığ kullanılıyor); `methodology_imbalance/{focal_loss_lin_2017,
class_balanced_loss_cui_2019,weighted_random_sampler_torch}` (hiç atıfsız);
`primary_baselines/{04_ahmed_2023_hybrid_fusion, 05_he_2025_hemf,
07_guo_2024_curriculum_ssl}` (dolu stub, kullanılmıyor). **Stub'ı OLMAYAN** (kullanılamaz):
`06_ramachandran_2025_interpretable`, `08_montalbo_2024_mfure_cnn`.

---

## Slice F — Yöntem derinleştirme (model tasarımı + implementasyon) — ✅ TAMAM (2026-06-25)

**Önem:** Major (model tasarımı %15 + implementasyon %20).
**Sonuç:** Omurga önyargıları (ViT global/minimal önyargı, Swin hiyerarşik/kaydırmalı
pencere/yerellik, BEiT maskeli görüntü modelleme/öz-denetimli) makalelerine dayalı
eklendi; transfer gerekçesi (catastrophic forgetting → yalnız son bloklar + LLRD
mantığı) yazıldı; rollout kalıntı terimi ($0{,}5\,I$) ve drop-path/MixUp/etiket
yumuşatma/EMA bir-cümleyle tanımlandı. Yeni atıf yok (5 stub zaten bib'de). Yöntem
516→713 kelime. Temiz derleme; sayı değişmedi.
**Kapsadığı bulgular:** Yöntem'deki sığ alt bölümler ve tanımsız terimler.

**Yapılacaklar:**
1. **Omurga tümevarımsal önyargıları** (`02_methodology.tex` §Omurgalar, ~42–48):
   her omurga için bir cümle, makalelerine dayalı — ViT küresel öz-dikkat / evrişim
   önyargısı yok~\cite{dosovitskiy2021vit}; Swin hiyerarşik + kaydırmalı pencere /
   yerel-çok ölçekli~\cite{liu2021swin}; BEiT maskeli görüntü modellemesiyle
   öz-denetimli~\cite{bao2022beit}. Bu, neden tamamlayıcı olduklarını gerekçelendirir
   (§5.5'e bağlanır).
2. **Transfer gerekçesi** (§Transfer, ~76–82): neden yalnız son bloklar + LLRD —
   küçük veride felakete-varan unutmayı (catastrophic forgetting) önlemek; LLRD'nin
   katman bazında düşük-LR mantığı~\cite{howard2018ulmfit}.
3. **Terim bir-cümle tanımları:** stokastik derinlik/drop-path, EMA, MixUp/etiket
   yumuşatma ne yapar; dikkat yayılımındaki kalıntı terimi ($0{,}5\,I$) neden eklenir
   (kalıntı bağlantılarını hesaba katma~\cite{abnar2020rollout}).

**Kabul:** Omurga alt bölümü her omurganın önyargısını açıklıyor; transfer seçimi
gerekçeli; kullanılan terimler okuyucunun ihtiyaç duyduğu yerde bir cümleyle tanımlı;
temiz derleme; sayı değişmez. **Veri:** ✅ (omurga + ulmfit + rollout stub'ları).
**Bağımlılık:** yok.

---

## Slice G — Tartışma genişletme + alt bölümler (§5.5, en kritik) — ✅ TAMAM (2026-06-25)

**Önem:** Major (en ağırlıklı bölüm; şu an 583 kelime).
**Sonuç:** İki yeni alt bölüm eklendi — "Hata analizi ve sınıf karışmaları" (ordinal
kolit-derecesi + anatomik hemorroids→retroflex-rectum / oesophagitis-a→normal-z-line
karışmaları, yalnız mevcut confusion matrix/per-class'tan) ve "Dengesizliğin makro-F1
üzerindeki etkisi" (Menon dengeli-hata + Lin 2017 focal; sampler prozada atıfsız).
Omurga-öznitelik tartışması frozen tabloya bağlandı. Tartışma 583→878 kelime, alt
bölüm 5→7. **Stub uyarısı doğrulandı:** `class_balanced_loss_cui_2019` stub'ı yanlış
içerik (Ahmed 2023) → **atıf verilmedi**; `weighted_random_sampler` paper.md yok →
atıfsız anlatıldı. Sadece doğrulanan focal (Lin 2017) bib'e eklendi. Temiz derleme;
yeni sayı yok.
**Kapsadığı bulgular:** Tartışma'nın inceliği; hata analizi ve dengesizlik
çözümlemesi eksik.

**Yapılacaklar:**
1. **Yeni alt bölüm "Hata analizi ve sınıf karışmaları"** (`05_discussion.tex`):
   mevcut karışıklık matrisi + sınıf bazında tablodan — ordinal karışmalar
   (ülseratif kolit dereceleri bitişik derecelere) ve anatomik karışmalar
   (`hemorroids` $\to$ `retroflex-rectum`, `oesophagitis-a` $\to$ `normal-z-line`)
   klinik olarak yorumlanır. **Yalnız mevcut figür/tablo; yeni sayı yok.**
2. **Yeni alt bölüm "Dengesizliğin makro-F1 üzerindeki etkisi"**: nadir sınıfların
   makro-F1'i neden sınırladığı, dengeli-hata çerçevesi~\cite{menon2020logitadjust}
   ile; odak kaybı (focal)~\cite{lin2017focal} ve sınıf-dengeli
   kayıp~\cite{cui2019classbalanced} gibi standart dengesizlik kaldıraçlarının
   bağlamı ve ağırlıklı örnekleyicinin~\cite{paszke2019sampler} bu projedeki rolü.
3. **Omurga-öznitelik tartışmasını derinleştir** (§Hangi omurga): Swin'in gücünü
   yerel doku/çok ölçek önyargısına, UMAP ayrışmasına ve sınıf-bazlı başarıma daha
   açık bağla.

**Kabul:** Tartışma'da hata analizi ve dengesizlik için ayrı alt bölümler var; §5.5'in
beş sorusu daha derin yanıtlı; iddialar figür/tablo/literatüre dayalı; yeni sayı yok;
temiz derleme. **Veri:** ✅ (karışıklık matrisi, per-class, imbalance stub'ları).
**Bağımlılık:** F ve H'den sonra (omurga + literatür çerçevesini kullanır).

---

## Slice H — İlgili Çalışmalar & literatür temellendirme — ✅ TAMAM (2026-06-25)

**Önem:** Major (rapor kalitesi %20; kullanıcı literatür derinliğini vurguladı).
**Sonuç:** Üç doğrulanmış çalışma eklendi — Ahmed 2023 (GI öznitelik füzyonu, CNN+
sınıflandırıcı), He 2025 (HEMF çok-dikkatli hiyerarşik füzyon), Guo 2024 (HiperKvasir
etiketsiz veriyle müfredat SSL; BEiT çizgisine bağlandı). Füzyon paragrafı tıbbi/GI
füzyonla derinleştirildi; GI paragrafına SSL eklendi. `references.bib`'e 3 entry.
İlgili Çalışmalar artık 9 farklı GI/comparator çalışmasına değiniyor. Temiz derleme;
SOTA yok. (focal G'de yapıldı; cui/sampler kullanılamaz olduğu için atıf verilmedi.)
**Kapsadığı bulgular:** sığ omurga/füzyon paragrafları; kullanılmayan dolu stub'lar;
imbalance literatürünün yokluğu.

**Yapılacaklar:**
1. **GI füzyon/SSL çalışmaları ekle** (`01b_related_work.tex` + `references.bib`):
   endoskopide birleşik-öznitelik melez modelleri~\cite{ahmed2023hybridfusion},
   çok-dikkatli hiyerarşik füzyon~\cite{he2025hemf}; öz-denetimli/müfredat
   ön-eğitim~\cite{guo2024curriculumssl} (BEiT'e bağlanır). Hepsi bağlamsal.
2. **Dengesizlik yöntemleri paragrafı/cümlesi:** odak kaybı, sınıf-dengeli kayıp ve
   ağırlıklı örnekleyicinin standart kaldıraçlar olduğu — Tartışma'daki (Slice G)
   çözümlemeyi besler.
3. **Omurga/füzyon paragraflarını derinleştir:** her omurganın ayırt edici özelliğini
   bir cümle genişlet.

**Kabul:** İlgili Çalışmalar ≥5 doğrudan-ilgili çalışmaya değiniyor (mevcut + yeni);
imbalance literatürü tanıtılmış; her yeni `\cite` non-empty stub'a eşli; SOTA yok;
temiz derleme. **Veri:** ✅ (04/05/07 + imbalance stub'ları).
**Risk / G sonrası güncelleme:** `focal` (Lin 2017) doğrulandı ve **Slice G'de**
atıflandı; `class_balanced_loss_cui_2019` stub'ı **yanlış içerik** (Ahmed 2023) ve
`weighted_random_sampler` **paper.md yok** → bu ikisi **kullanılmaz** (atıf verme).
Dolayısıyla H artık esas olarak GI füzyon/SSL çalışmalarına (Ahmed `04`, He `05`,
Guo `07`) odaklanır; `07_guo` stub içeriği atıftan önce doğrulanmalı.
**Bağımlılık:** yok.

---

## Slice I — Giriş & Sonuç güçlendirme + terim tanımları — ✅ TAMAM (2026-06-25)

**Önem:** Minor.
**Sonuç:** Giriş'te üç-omurga seçimi tamamlayıcı önyargılarla (küresel / hiyerarşik
yerel / öz-denetimli) gerekçelendirildi; Sonuç gelecek-iş satırına odak kaybı
(focal)~\cite{lin2017focal} atfı eklendi (ordinal/ince-taneli jenerik kaldı — stub
yok); Deneyler'de ağırlıklı örnekleyici (ters-frekans örnekleme) ve önyükleme
(bootstrap) bir-cümleyle tanımlandı. Yeni atıf yok (focal G'de eklenmişti). Temiz
derleme; sayı değişmedi.
**Kapsadığı bulgular:** Giriş'te backbone-seçim gerekçesi sığ; Sonuç kısa/atıfsız;
Deneyler'de sampler/bootstrap tanımsız.

**Yapılacaklar:**
1. **Giriş** (`01_introduction.tex`): üç omurga seçiminin gerekçesini (tamamlayıcı
   önyargılar) bir-iki cümleyle güçlendir (Slice F ile tutarlı).
2. **Sonuç** (`06_conclusion.tex`): gelecek-iş satırını ilgili literatüre bağla —
   sıralı (ordinal) kayıplar + odak/sınıf-dengeli kayıp~\cite{lin2017focal,
   cui2019classbalanced}.
3. **Deneyler** (`03_experimental_setup.tex`): ağırlıklı örnekleyici (ters-frekans
   örnekleme) ve önyükleme (bootstrap) GA prosedürü için birer cümle tanım.

**Kabul:** Giriş'te omurga gerekçesi açık; Sonuç gelecek-işi atıflı; sampler/bootstrap
tanımlı; temiz derleme; sayı değişmez. **Veri:** ✅. **Bağımlılık:** F, H sonrası.

---

## Slice J — Literatür metrik karşılaştırmasını zenginleştir (Sonuçlar §karşılaştırma) — ✅ TAMAM (2026-06-25)

**Önem:** Major (rapor kalitesi %20; kullanıcı metrik karşılaştırmasının zayıf
olduğunu belirtti).
**Sonuç:** `tab:context` 6 sütunlu (Çalışma/Yöntem/Protokol/Doğruluk/Makro-F1/Diğer)
zengin tabloya çevrildi; **Borgli macro-F1 0.619** (micro 0.907, MCC 0.899) çıpa
olarak eklendi; rakiplerin macro-F1'i "$-$" (GastroViT 0.920, Wang 0.868, EffiMix
0.980 doğruluk + ağ. F1 0.97 notu). İki `\subsubsection` (Protokol ve ölçüt farkları
/ Metrik konumlandırma); 0.612'nin Borgli CNN bandında (0.605–0.619) olduğu bağlamsal
gösterildi. Tüm sayılar paper.md'den doğrulandı; SOTA yok. 04_results 1176→1317 kelime.
**Amaç:** `tab:context`'i dürüst, bağlamsal ama daha zengin bir tabloya çevirmek;
özellikle veri kümesi makalesinin (Borgli) **macro-F1** çıpasını eklemek.

**Literatür madenciliği (paper.md'lerden doğrulanmış):**
- **Borgli 2020 (`03`) — KİLİT:** HyperKvasir veri kümesi makalesi, iki-bölme ortalaması,
  CNN; **macro-averaged** P/R/F1 + micro + MCC raporlar. En iyi: DenseNet-161
  **macro-F1 0.619** (macro-P 0.640 / macro-R 0.616), micro-F1 0.907, MCC 0.899;
  ResNet-152 macro-F1 0.606; ortalama topluluk macro-F1 0.617. **Bizim 0.612 macro-F1
  bu CNN baseline bandında** ve macro(0.62)–micro(0.91) uçurumu aynı dengesizlik
  olgusunu gösterir.
- **GastroViT 2025 (`02`):** HyperKvasir 23-sınıf (10.662), MobileViT topluluğu;
  doğruluk **91.98%**. Karşılaştırılabilir leakage-free **macro-F1 raporlamaz → "$-$"**.
- **EffiMix 2022 (`01`):** HyperKvasir; doğruluk **97.99%**, F1 **0.97 (genel/ağırlıklı,
  macro değil)** → macro-F1 "$-$", not olarak ağırlıklı F1.
- **Wang 2023 (`09`):** HyperKvasir; doğruluk **86.81%**; macro-F1 raporlanmaz → "$-$".
- **Guo 2024 (`07`):** HyperKvasir, SSL; accuracy + F1 raporlar (güçlü omurgalarda
  ~%85 F1) ama **F1 türü belirsiz** → tabloya alınmaz; istenirse prozada bağlamsal.
- Dışlananlar: Ahmed (`04`, Kvasir), He (`05`, genel medikal), Subedi/Varam
  (`10`/`11`, Kvasir-Capsule) → comparator tablosuna girmez (farklı veri kümesi).

**Yapılacaklar:**
1. `tab:context`'i genişlet — sütunlar: **Çalışma | Yöntem (omurga/füzyon) | Protokol |
   Doğruluk | Makro-F1 | Diğer (mikro-F1/MCC veya not)**. Borgli satırını **macro-F1
   0.619** (+ MCC 0.899, micro 0.907) ile ekle; GastroViT/Wang/EffiMix doğruluklarını
   koru, macro-F1 hücreleri "$-$" + dipnot ("doğrudan karşılaştırılabilir macro-F1
   bildirilmemiş"); EffiMix notunda "ağırlıklı F1 0.97".
2. Karşılaştırmayı iki `\subsubsection`'a böl: **Protokol farkları** ve **Metrik
   konumlandırma**; ikincisinde 1-2 cümle bağlamsal konum — bizim macro-F1'imizin
   veri kümesi makalesinin CNN baseline bandında olduğu, yüksek-doğruluk değerlerinin
   farklı protokol + toplu/ağırlıklı metriklerden kaynaklandığı (SOTA iddiası yok).

**Kabul:** `tab:context` zenginleşmiş ve her hücre `paper.md`'ye izlenebilir veya
"$-$"; Borgli macro-F1 çıpası var; iki `\subsubsection`; SOTA yok; temiz derleme.
**Veri:** ✅ (Borgli/GastroViT/EffiMix/Wang doğrulandı). **Bağımlılık:** yok.

## Slice J-2 — Destekleyici ölçütler (MCC + weighted-F1) ile karşılaştırmayı zenginleştir — ✅ TAMAM (2026-06-25)

**Önem:** Major (deneysel analiz %20; Borgli ile elma-elmaya kıyas).
**Sonuç:** Pooled OOF (n=10662) tahminlerinden türetildi: doğruluk **0.878**
(=micro-F1), **weighted-F1 0.879**, **MCC 0.868**; macro-F1 0.6119 cross-check ile
doğru pooling teyit edildi. `tab:context` bizim satıra **MCC 0.868** eklendi (Borgli
MCC 0.899 ile aynı eksen); §5.4'e destekleyici-ölçüt paragrafı (macro-F1 headline
kalır, CI yalnız macro-F1'de). Dürüst nüans: MCC Borgli'nin biraz altında ama aynı
bantta. Uydurma yok; temiz derleme.
**Amaç:** Şampiyonun **mevcut** pooled OOF tahminlerinden ek standart ölçütler
**türetmek** (yeni deney/A100 yok, uydurma yok) ve karşılaştırmayı Borgli'nin
raporladığı ölçüt setine (macro-F1 + micro-F1 + MCC) yaklaştırmak.

**Gerekçe:** Borgli macro-F1 0.619 \emph{ve} micro-F1 0.907 \emph{ve} MCC 0.899
raporluyor; biz şu an yalnız macro-F1 + doğruluk gösteriyoruz. MCC dengesizliğe
dayanıklı tek-sayı ölçüttür ve doğrudan karşılaştırılabilir. micro-F1, tek-etiketli
çok-sınıfta **doğruluğa eşittir** → ayrı boyut değil, not düşülür.

**Yapılacaklar:**
1. `11_triple_weighted_cv*/predictions.npz` (5 fold birleşik, n=10662) üzerinden
   **MCC** (`sklearn.matthews_corrcoef`) ve **weighted-F1** (`f1_score(average='weighted')`)
   hesapla; pooled doğruluk = micro-F1 olduğunu doğrula. (Küçük hesap; `summarize_vit_cv`
   yardımcılarını veya `make_vit_report_figures.py`'ı kullan — model yüklemeye gerek yok.)
2. `tab:context`'te **bizim satırın "Diğer" hücresine MCC** ekle (Borgli'nin MCC 0.899
   sütununa karşılık gelir).
3. Sonuçlar §5.4'e tek cümlelik **destekleyici ölçüt** notu (MCC + weighted-F1),
   macro-F1 headline kalır; CI yalnız macro-F1'de.

**Kabul:** MCC + weighted-F1 pooled OOF'tan hesaplanmış ve `predictions.npz`'e
izlenebilir; `tab:context`'te bizim MCC, Borgli MCC ile aynı eksende; macro-F1
headline (VLD-10) korunur; uydurma sayı yok; temiz derleme. **Veri:** ✅
(predictions.npz mevcut). **Bağımlılık:** J sonrası, K öncesi.

## Slice K — "Güçlü ve zayıf yönler"i derinleştir + alt-alt başlıklar — ✅ TAMAM (2026-06-25)

**Önem:** Major (Tartışma en ağırlıklı; bölüm şu an sığ).
**Sonuç:** İki `\subsubsection` (Güçlü yönler / Sınırlamalar) madde-madde kanıta
bağlandı (McNemar Bölüm~ref, Borgli bandı + MCC `tab:context`, CI `tab:cv`,
yorumlanabilirlik figürleri, negatifler `tab:addons`/`fig:logit`, throughput
`tab:cost`; sınırlamalar `tab:perclass` nadir-sınıf F1=0, ordinal/anatomik karışma,
CI-örtüşen kazanç, BEiT/GMU, yaklaşık üreme, tek veri kümesi). Yeni sayı yok; tüm
`\ref`'ler çözüldü. Tartışma 878→1068 kelime; temiz derleme.
**Amaç:** Tek bloğu kanıta-bağlı `\subsubsection{Güçlü yönler}` ve
`\subsubsection{Sınırlamalar}` olarak derinleştirmek.

**Kanıta bağlı güçlü yönler:** (i) çok-ViT füzyonu en iyi tekli/ikiliden **istatistiksel
anlamlı** daha doğru (McNemar p<1e-4); (ii) sızıntıya-karşı resmi 5-katlı + bootstrap CI
(titiz değerlendirme); (iii) macro-F1 veri kümesi makalesinin CNN baseline bandında
(Borgli 0.62 — Slice J); (iv) yorumlanabilirlik klinik bölgelerle uyumlu (rollout/
Grad-CAM++/UMAP); (v) dürüst negatifler (GMU/TTA/seed/logit-adj); (vi) verimlilik
ayarı ~5–6×.
**Kanıta bağlı sınırlamalar:** (i) nadir-sınıf tavanı (hemorroids 6 / UC-1-2 11 → F1≈0);
(ii) ordinal/anatomik karışmalar; (iii) macro–accuracy uçurumu; (iv) füzyonun macro-F1
kazancı CI düzeyinde sınırlı (anlamlılık doğruluk düzeyinde); (v) tek başına BEiT zayıf,
GMU kazanç yok; (vi) yaklaşık üreme (bf16/TF32); (vii) tek veri kümesi, klinik/dış
doğrulama yok.

**Yapılacaklar:** `05_discussion.tex` §Güçlü ve zayıf yönler'i iki `\subsubsection`'a
böl; her maddeyi mevcut kanıta (tablo/figür/sonuç) bağla; yeni sayı yok.
**Kabul:** iki alt-alt başlık; her madde kanıta izlenebilir; bölüm belirgin
derinleşmiş; temiz derleme. **Veri:** ✅. **Bağımlılık:** Slice J (Borgli çıpası).

## Slice L — Bölümlere `\subsubsection` granülerliği (kalan bölümler) — ✅ TAMAM (2026-06-25)

**Önem:** Minor.
**Sonuç:** Mevcut proza yeniden yuvalandı (yeni içerik/sayı yok): Yöntem §İzdüşüm-
füzyon-sınıflandırıcı → 3 alt-alt başlık (Dal izdüşümü / Füzyon yöntemleri /
Sınıflandırıcı); Sonuçlar §Nihai modelin ayrıntılı çözümlemesi → 2 (Eğitim dinamiği /
Karışıklık matrisi ve sınıf bazında başarım); Tartışma §Yorumlanabilirlik → 2 (Dikkat
ve Grad-CAM++ / Öznitelik uzayı (UMAP)). Tüm `\label`/`\ref` korundu (0 undefined ref);
temiz derleme.
**Amaç:** J/K dışındaki bölümlerde tutarlı derin yapı.
**Yapılacaklar (mevcut metni böl, label'ları koru):**
- Yöntem §İzdüşüm, füzyon ve sınıflandırıcı → `\subsubsection{Dal izdüşümü}` /
  `\subsubsection{Füzyon yöntemleri}` / `\subsubsection{Sınıflandırıcı}`.
- Sonuçlar §Nihai modelin ayrıntılı çözümlemesi → `\subsubsection{Eğitim dinamiği}` /
  `\subsubsection{Karışıklık matrisi ve sınıf bazında başarım}`.
- Tartışma §Yorumlanabilirlik → `\subsubsection{Dikkat ve Grad-CAM++}` /
  `\subsubsection{Öznitelik uzayı (UMAP)}`.
**Kabul:** ilgili `\subsection`'lar anlamlı `\subsubsection` çocukları kazanır;
mevcut `\label`/çapraz-referanslar bozulmaz; içerik/sayı değişmez; temiz derleme.
**Veri:** ✅. **Bağımlılık:** J, K sonrası.

---

## Slice M — Kontrollü CNN vs ViT karşılaştırması (yeni alt bölüm) — ✅ TAMAM (2026-06-25)

**Önem:** Major (deneysel analiz %20 + tartışma; raporun tek elma-elmaya kıyası).
**Sonuç:** Tartışma'ya `\subsection{Aynı protokol altında CNN füzyonu ile karşılaştırma}`
(`tab:cnnvit`) eklendi. CNN sayıları `docs/FINAL_MODEL.md`'den **bu turda doğrulandı**:
CNN nihai (üçlü ağırlıklı + **TTA**) macro-F1 0.6075 [0.5860,0.6296] / acc 0.8765 /
MCC 0.8662; CNN base-CE 0.6000 (ViT CI içinde). ViT (base) 0.6119 / 0.8779 / 0.868.
Dürüst okuma: **istatistiksel denk**, ortak nadir-sınıf tavanı, üstünlük/SOTA yok;
ortak negatifler (focal/TTA/seed) + çapraz-proje yönelimler (dönüştürücü omurgalar +
yorumlanabilirlik uygulandı). §9 temiz (gövdede path/VLD/sprint yok). 05_discussion
1068→1367 kelime; temiz derleme.
**Amaç:** Önceki CNN füzyon projesinin sonuçlarını, **aynı veri kümesi / resmi 5-katlı
protokol / leakage-free OOF+CI / macro-F1 headline / aynı füzyon çerçevesi** altında
ViT füzyonuyla kıyaslayan kontrollü bir bölüm eklemek (VLD-14 ile tasarlanmış kıyas).

**Veri (yazım turunda CNN artefaktlarından doğrulanacak — hafızadan değil):**
- ViT (otoriter): macro-F1 **0.6119**, doğruluk **0.878**, MCC **0.868** (base; TTA
  kazanç vermedi).
- CNN (`docs/FINAL_MODEL.md` + CNN sonuç tabloları): üçlü ağırlıklı füzyon + MLP, aynı
  resmi 5-katlı; pooled macro-F1 ~**0.6075** (TTA) / ~**0.6000** (CE base), doğruluk
  ~**0.8765**, MCC ~**0.8662**. **Hangi sayının base/TTA olduğu açıkça belirtilecek;**
  adil kıyas için ViT-base ↔ CNN-base önerilir. Dokümante olmayan CNN metriği "$-$".

**Yapılacaklar:** `05_discussion.tex`'e yeni `\subsection{Aynı protokol altında CNN
füzyonu ile karşılaştırma}` (label'lar korunur; alt bölüme köprü cümlesi). Üç eksen:
1. **Metrik karşılaştırması (aynı protokol):** küçük CNN-vs-ViT tablosu (macro-F1
   headline + doğruluk + MCC); her sayının inference reçetesi (ViT base; CNN base-CE
   ve/veya TTA) belirtilir. Dürüst okuma: istatistiksel olarak **denk** (CI örtüşür),
   her ikisi de aynı nadir-sınıf tavanına çarpar; ViT marjinal yüksek ama anlamlı değil.
   Üstünlük/SOTA iddiası YOK.
2. **Ortak negatif bulgular:** her iki projede de kazanç vermeyen teknikler (focal,
   TTA, seed topluluğu) → mimari-ailesinden bağımsız veri sınırı.
3. **Çapraz-proje yönelimler:** CNN projesinin gelecek-iş maddelerinden ViT'te
   gerçekleştirilenler (dönüştürücü omurgalar, dikkat/Grad-CAM++ yorumlanabilirlik) ve
   her ikisinde terk edilenler (yalnız dokümanlıysa).

**Kabul:** Kontrollü kıyas tablosu (aynı protokol vurgulu), her CNN sayısı CNN
artefaktına izlenebilir + reçetesi belirtilmiş; metodolojik çerçeve (self-referential
proje-tarihi değil); gövdede repo-path/VLD-PLD/sprint-hafta ifadesi yok; macro-F1
headline; SOTA yok; temiz derleme. **Veri:** ✅ (CNN dokümante; doğrulanacak).
**Bağımlılık:** J-2 sonrası (ViT MCC çıpası), K ile uyumlu.

---

## Özet matris & sıra

| Slice | Odak | Önem | Veri | Bağımlılık | Durum |
|---|---|---|---|---|---|
| F — Yöntem derinleştirme | omurga önyargıları + transfer gerekçesi + terimler | Major | ✅ | — | ✅ Tamam |
| G — Tartışma genişletme | hata analizi + dengesizlik alt bölümleri | Major | ✅ | F, H | ✅ Tamam |
| H — Literatür temellendirme | GI füzyon/SSL (Ahmed/He/Guo) | Major | ✅ | — | ✅ Tamam |
| I — Giriş/Sonuç + terimler | gerekçe + gelecek-iş atıfları + tanımlar | Minor | ✅ | F, H | ✅ Tamam |
| J — Literatür metrik karşılaştırması | zengin tab:context + Borgli macro-F1 çıpası | Major | ✅ | — | ✅ Tamam |
| J-2 — Destekleyici ölçütler | MCC 0.868 + weighted-F1 0.879 (pooled OOF) | Major | ✅ | J | ✅ Tamam |
| K — Güçlü/zayıf derinleştirme | kanıta-bağlı + 2 alt-alt başlık | Major | ✅ | J, J-2 | ✅ Tamam |
| L — \subsubsection granülerliği | Yöntem/Sonuçlar/Tartışma derin yapı | Minor | ✅ | J, K | ✅ Tamam |
| M — Kontrollü CNN vs ViT | aynı-protokol metrik + ortak negatifler + çapraz-yönelim | Major | ✅ | J-2 | ✅ Tamam |

**Önerilen sıra (deadline Cuma 2026-06-26):** F → H → G → I → J → J-2 → K → L (tamam)
→ **M** (kontrollü CNN vs ViT karşılaştırması — ViT MCC çıpasını kullanır).

**Genel not:** Tüm geliştirmeler **yazımı** derinleştirir, sonuçları değil. Yeni
sayı/atıf uydurulmaz; atıflanacak her stub içeriği önce doğrulanır. Stub'ı olmayan
çalışmalar (Ramachandran, Montalbo) kullanılmaz.
