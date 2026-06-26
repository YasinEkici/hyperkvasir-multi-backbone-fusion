# ViT Raporu — Analiz ve Düzeltmeler Planı (Sprint 5, Slice 4 sonrası)

Bu dosya, `reports/vit/` altındaki çoklu-ViT füzyon raporunun tam denetiminden
(2026-06-24/25) çıkan bulguları, **birbirine bağlı düzeltme dilimlerine (slice)**
gruplar. Amaç: rapor taslağını teslime hazır hale getirmek. Kaynak doğruluk:
`docs/report/AGENT_REPORT_RULES.md` (yazım kuralları), `docs/vit/final_assignment.md`
§5 (rubrik), `docs/vit/project_plan.md` §10–§11.

**Kapsam kuralları (tüm dilimler):**
- Inference-only zihniyet: **yeni eğitim yok, 0 A100**. Her düzeltme mevcut
  artefaktlardan (`results/vit/`, `reports/vit/figures/`) veya figür/tabloyu
  inference-free yeniden üreterek yapılır. **Sayı uydurulmaz.**
- Headline ölçüt makro-F1; nihai model **dondurulmuş** `11_triple_weighted` —
  hiçbir kilitli karar, şampiyon veya Sprint 4/4.5 sayısı değiştirilmez.
- Rapor dili akademik Türkçe; "seed" ("tohum" değil); rapor **gövdesinde** repo
  yolu / iç karar kimliği (VLD-*) / self-referential "dürüstlük" ifadesi olmaz;
  SOTA iddiası yok; literatür yalnızca bağlamsal.
- LaTeX hedefi pdfLaTeX/Overleaf (yerelde `tectonic` ile de derlenir). Her dilim
  sonrası `cd reports/vit && tectonic main.tex` ile temiz derleme doğrulanır
  (0 undefined citation/ref, 0 missing Türkçe glyph).
- Onay-kapılı: dilimler sırayla, her biri ayrı onayla.

**Önerilen dilim sırası (deadline Cuma 2026-06-26):** A → B → C → D → E.

---

## Slice A — Yazım kuralları & stil uyumu (gövde geneli) — ✅ TAMAM (2026-06-25)

**Kapsadığı bulgular:** S1, S2, S3, S4, S5
**Sonuç:** Gövdede repo-path 0, "dürüst" 0, VLD-id 0, dekoratif em-dash 0; yasak
"yalnızca…değil" 0; rapor temiz derlendi (0 undefined cite/ref, 0 missing Türkçe
glyph). Sayılar/sonuçlar değişmedi.
**Amaç:** AGENT_REPORT_RULES §2/§9 ihlallerini gidermek; metni saflaştırmak.
Negatif sonuçların *kendisi korunur*, yalnızca self-referential çerçeveleme
sadeleştirilir.

**Yapılacaklar:**
1. **S1 — Gövdeden repo yollarını kaldır** (§9): `03_experimental_setup.tex:44`
   (`results/vit/runs/{id}/metrics.json`), `04_results.tex:17/46/114` (tablo
   başlıklarındaki "Kaynak: results/vit/tables/…"), `06_conclusion.tex:29`
   (`results/vit/`). Yerine genel ifade ("değerlendirme çıktılarından izlenebilir")
   veya başlıktan tamamen çıkar.
2. **S2 — "dürüst negatif" self-referential ifadesini düz ifadeye çevir** (§9 /
   report-style): `01_introduction.tex:37`, `04_results.tex:177`,
   `05_discussion.tex:36`, `06_conclusion.tex:16`. Örn. "GMU, ağırlıklı füzyonu
   geçememiştir" — sonucu koru, "dürüst negatif olarak raporlanmıştır" ifadesini at.
3. **S3 — Dekoratif em-dash (`---`) kullanımını azalt** (§2): `04_results.tex` (5×),
   `05_discussion.tex` (3×). Virgül/iki nokta/parantezle değiştir; yapısal
   parantezlikler kalabilir.
4. **S4 — "yalnızca" manuel teyit** (§2): 9 kullanım var; hiçbiri yasak
   "yalnızca…değil" kalıbı değil — okuyup gereksiz olanları sadeleştir.
5. **S5 — §5.1 "3 sınıflandırıcı" ele alışını tek cümlede tut**: sınıflandırıcının
   sabit MLP olduğu ve karşılaştırmanın füzyon/omurga/transfer ekseninde yapıldığı
   mevcut ifade korunur; genişletilmez.

**Kabul:** Gövdede repo yolu / VLD-id / "dürüst" ifade kalmaz; em-dash dekoratif
kullanımı temizlenir; rapor temiz derlenir; sayılar/sonuçlar değişmez.
**Veri:** ✅ hazır (yalnız metin). **Bağımlılık:** yok.

---

## Slice B — Eksik zorunlu tablolar + eğitim süresi (§5.3–§5.4, §5.5) — ✅ TAMAM (2026-06-25)

**Kapsadığı bulgular:** R1, R2, R3
**Sonuç:** R1 frozen ablation tablosu (11 satır, `tab:frozen`), R3 hiperparametre
tablosu (`tab:hparams`), R2 transfer-maliyet tablosu (`tab:cost`, en iyi epoch +
~5.3×/5.9× verim) eklendi; §5.5 süre sorusu yanıtlandı. **Not (R2):** duvar-saati
loglanmadığından maliyet, yakınsama epoch'u (history'den) + ayarlı verimle ifade
edildi — uydurma sayı yok. Temiz derleme (0 undefined cite/ref, 0 missing glyph).
**Amaç:** §10'un 4 zorunlu tablosundan eksik olanları tamamlamak ve §5.5'in
"hangi transfer ne kadar sürmüş?" sorusunu yanıtlamak.

**Yapılacaklar:**
1. **R1 — Frozen ablation tablosu** (§10 tablo #1): `frozen_ablation_ranked.md`
   (11 satır, 0. kat) verisinden bir tablo ekle (en az ilk 6 + "tam liste depoda"
   notu, ya da tam 11 satır). `04_results.tex` "Dondurulmuş eleme" alt-bölümüne.
2. **R2 — Training-time / transfer karşılaştırma tablosu + §5.5 süre paragrafı**
   (§10 tablo #4): **önce** `results/vit/runs/11_triple_weighted_cv*/epoch_log.jsonl`
   ve `02_single_swin_t_cv*` zaman damgalarını kontrol et. Gerçek duvar-saati varsa
   onu kullan; yoksa throughput (~5.3× tekli / ~5.9× üçlü, ~55 img/s) + epoch
   sayısından **türetilmiş** (uydurma değil) yaklaşık süre tablosu ver. `05_discussion`
   transfer alt-bölümüne süreyi açıkça yaz.
3. **R3 — Hiperparametre tablosu** (§5.3): `configs/vit/training/vit_finetune.yaml`
   değerlerinden (AdamW, backbone_lr 5e-5, head_lr 1e-3, LLRD 0.7, wd 0.05, warmup 5,
   drop_path 0.05, label smoothing 0.1, mixup 0.2, EMA 0.9998, batch 32, unfreeze 3)
   `03_experimental_setup.tex`'e tablo.

**Kabul:** §10'un 4 tablosu da raporda; §5.5 süre sorusu yanıtlı; her sayı
`metrics.json`/`epoch_log`/config'e izlenebilir; temiz derleme.
**Veri:** R1 ✅, R3 ✅, **R2 ⚠ (epoch_log zaman damgası kontrolü gerekli)**.
**Bağımlılık:** Slice A önce (aynı dosyalara dokunur).

---

## Slice C — Literatür derinliği & atıf bütünlüğü (§5.1/İlgili Çalışmalar, §11) — ✅ TAMAM (2026-06-25)

**Kapsadığı bulgular:** L1, L2, L3
**Sonuç:** L2 — `selvaraju2017gradcam` ve `howard2018ulmfit` stub'ları oluşturuldu
(artık her `\cite` non-empty stub). L1 — İlgili Çalışmalar Subedi 2024 (CNN-Transformer
melez) ve Varam 2024 (hafif/edge ViT) ile derinleştirildi (bağlamsal; `references.bib`
genişletildi). L3 — comparator sayıları `paper.md`'lerle teyit edildi: GastroViT aynı
veri kümesi (~%92 doğruluk), EffiMix ~%98 (ağırlıklı F1); rakiplerin doğrudan
karşılaştırılabilir makro-F1'i olmadığından tabloda "$-$". **Ramachandran 2025
düşürüldü** (paper.md yok → §11 ihlali olur). Temiz derleme; SOTA iddiası yok.
**Amaç:** İlgili çalışmaları derinleştirmek ve §11 "her atıfın stub'ı olmalı"
kuralını sağlamak.

**Yapılacaklar:**
1. **L2 — Eksik 2 stub'ı oluştur** (§11): `selvaraju2017gradcam` →
   `references/methodology_evaluation/gradcam_selvaraju_2017/paper.md`;
   `howard2018ulmfit` (LLRD) → uygun bir `references/methodology_training/…/paper.md`.
   Slice 1-3'teki stub formatında (kısa, neden atıfta, yöntem özeti).
2. **L1 — İlgili çalışmaları derinleştir**: kullanılmayan doğrudan-ilgili stub'lardan
   yararlan — `10_subedi_2024_cnn_swin_fusion` (CNN+Swin füzyon),
   `11_varam_2024_edge_vits_capsule` (edge ViT), `06_ramachandran_2025_interpretable`
   (yorumlanabilir HiperKvasir). `01b_related_work.tex`'e 2-3 cümle/paragraf;
   istenirse comparator tablosuna (bağlamsal) satır. `references.bib`'e yeni entry.
3. **L3 — Comparator sayılarını teyit et**: `tab:context`'teki Wang (~86.81 acc),
   GastroViT (~91.98 acc / ~0.64 F1), EffiMix değerlerini ilgili `paper.md`'lerle
   karşılaştır; tutmuyorsa düzelt; hepsi "bağlamsal, protokol farklı" çerçevesinde.

**Kabul:** Her `\cite` anahtarının non-empty stub'ı var; ilgili çalışmalar bölümü
en az 3 doğrudan-ilgili çalışmaya değiniyor; comparator sayıları doğrulanmış;
SOTA iddiası yok; temiz derleme (0 undefined citation).
**Veri:** ✅ hazır (stub'lar dolu/oluşturulabilir). **Bağımlılık:** yok (B'den bağımsız).

---

## Slice D — Yöntem rigoru & kanıt zenginleştirme — ✅ TAMAM (2026-06-25)

**Kapsadığı bulgular:** M1, R4, D1
**Sonuç:** M1 — birleştirme/ağırlıklı/GMU füzyon denklemleri Yöntem'e eklendi
(implementasyona sadık: ağırlıklı = `softmax(w)` skaler, GMU = öğe-bazında kapı).
R4 — tam 23-satır sınıf bazında tablo Ek A'ya eklendi. D1 — Ek B'de iki ek
yorumlanabilirlik paneli (rollout `normal-pylorus` + Grad-CAM++ `hemorroids`).
`main.tex`'e `\appendix` + `07_appendix.tex` bağlandı. Temiz derleme; PDF ~1.8 MB.
**Amaç:** "Model tasarımı %15" ve "deneysel analiz %20" için raporu güçlendirmek;
üretilmiş ama kullanılmayan kanıtları değerlendirmek.

**Yapılacaklar:**
1. **M1 — Füzyon denklemleri** (§5.2): `02_methodology.tex`'e ağırlıklı füzyon ve
   GMU (öğe-bazında kapı, $h=\sum_i z_i\odot h_i$; N=2 için bimodal indirgeme)
   denklemlerini ekle.
2. **R4 — Tam sınıf-bazında tablo**: `per_class_champion.md` (23 satır) tablosunu
   bir ek (appendix) olarak ekle; ana metindeki 6-satır özet kalır.
3. **D1 — Yorumlanabilirlik panel galerisi**: kullanılmayan 10 panelden (5 rollout +
   5 gradcam) küçük bir grid/appendix figürü oluştur (yaygın + nadir sınıflar);
   üretilmiş kanıtı değerlendir.

**Kabul:** Yöntemde füzyon denklemleri var; tam per-class tablo ekte; yorumlanabilirlik
kanıtı genişletilmiş; figür/tablo sağlamaları izlenebilir; temiz derleme.
**Veri:** ✅ hazır. **Bağımlılık:** Slice A/B/C sonrası (cila aşaması).

---

## Slice E — Teslim & final (kullanıcıya bağlı) — ✅ büyük ölçüde TAMAM (2026-06-25)

**Kapsadığı bulgular:** G1, G2
**Sonuç:** G1 — kapak tek yazar olarak güncellendi (Yasin Ekici, No 21360859029).
Kapağa ve Sonuç bölümüne GitHub deposu linki eklendi
(`https://github.com/YasinEkici/hyperkvasir-multi-backbone-fusion`); rapordaki
video `\TODO`'su kaldırıldı (video rapora değil, depo README'sine konacak).
Raporda gerçek `\TODO` kalmadı. **Kalan (kullanıcı tarafı):** Sprint 5 tanıtım
videosunun çekilip README'ye eklenmesi.
**Amaç:** Submission-blocker'ları kapatmak.

**Yapılacaklar:**
1. **G1 — Öğrenci ad/no**: `main.tex:48` `\TODO`'larını doldur (kullanıcı sağlar).
2. **G2 — YouTube linki**: Slice 5 videosu çekildikten sonra `06_conclusion.tex`
   ve README'ye linki yerleştir (kullanıcı sağlar).

**Kabul:** `\TODO` kalmaz; §11 teslim checklist'i tam; final PDF derlenir.
**Veri:** Kullanıcı. **Bağımlılık:** G2 ← Sprint 5 Slice 5 (video).

---

## Özet matris

| Slice | Bulgular | Önem | Veri | Bağımlılık | Durum |
|---|---|---|---|---|---|
| A — Stil/kural | S1,S2,S3,S4,S5 | Major+Minor | ✅ | — | ✅ Tamam |
| B — Tablolar+süre | R1,R2,R3 | Major | R2 ⚠ | A | ✅ Tamam |
| C — Literatür+stub | L1,L2,L3 | Major | ✅ | — | ✅ Tamam |
| D — Yöntem+kanıt | M1,R4,D1 | Minor | ✅ | A,B,C | ✅ Tamam |
| E — Teslim | G1,G2 | Blocker | Kullanıcı | Slice 5 | ✅ Rapor tarafı tamam (video README'ye kalan) |

Her dilim sonrası: `tectonic main.tex` temiz derleme + `docs/vit/results_progress.md`
güncellemesi. Hiçbir dilim kilitli kararları, şampiyonu veya mevcut sayıları
değiştirmez.
