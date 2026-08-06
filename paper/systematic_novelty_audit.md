# Audit systematique de nouveaute

Article audite: `Net-Aware Learned Block Mapping for Reversible Data Hiding in Binary Images`

Date de recherche: 2026-08-03

## Verdict utilisable contre un rejet "lack of novelty"

La nouveaute defendable n'est pas le block mapping 3x3, ni la RDH binaire, ni la notion generale de payload net. Ces elements existent deja dans des familles voisines.

La nouveaute defendable est la combinaison suivante, dans le sous-domaine precis de la RDH plaintext en images binaires par block mapping PEAK/ZERO:

1. un critere operationnel de capacite utile `C_net = C_gross - |A_wire|_2`;
2. un format auxiliaire executable et mesure en bits, avec codec enumeratif pour le sous-ensemble PEAK actif et les positions de flips;
3. une selection de prefixe de tables par maximisation du payload utile, pas par capacite brute;
4. un ordre des ZERO distance-one appris par CNN et valide hors image;
5. une comparaison commune PPOCP / Dong / Huynh-Nguyen / ABM avec memes covers, memes messages, memes cibles, DRD, reversibilite exacte, cout auxiliaire, statistiques appariees, ressources et steganalyse groupee;
6. une validation externe sur BOSSbase binarise sans re-entrainement.

Position ferme: l'article ne doit pas etre vendu comme "nouveau block mapping"; il doit etre vendu comme "premier cadre net-capacity/wire-level executable pour le block mapping RDH binaire appris et compare sous protocole commun". Sous cette formulation, aucun travail identifie ne fournit la meme combinaison.

## Protocole

### Bases et sources interrogees

- Bibliographie locale: `paper/references.bib`, `paper/main.tex`.
- OpenAlex API: recherche systematique, DOI lookup, backward/forward citation chasing.
- Crossref API: verification bibliographique par requetes bibliographiques simples.
- Pages editeur/DOI: Springer, ScienceDirect, MDPI, IEEE/DBLP/ACM quand disponibles.
- DBLP: verification des notices et liens "ask others" vers Google Scholar / Semantic Scholar.
- Web indexe: requetes exact-title et DOI pour completer les pages editeur bloquees.

Limites documentees:

- Google Scholar n'offre pas d'API publique stable et les pages directes n'ont pas donne de resultats exploitables par l'outil web. Les liens Google Scholar exposes par Springer/MDPI/DBLP ont ete verifies comme points de depart, mais le citation chasing reproductible a ete fait via OpenAlex et Crossref.
- Semantic Scholar public a renvoye un rate-limit HTTP 429 lors d'un essai precedent. La cle API fournie dans la conversation n'a pas ete reutilisee dans une commande shell pour eviter de la reexposer dans les journaux. Les resultats Semantic-Scholar-like ont donc ete approximes par OpenAlex, Crossref, DBLP et pages editeur.
- Crossref renvoie des totaux tres larges sur les requetes bibliographiques libres; les top results et les DOI ont ete utilises comme verification, pas comme comptage d'inclusion final.

### Criteres d'inclusion

Inclus:

- RDH ou data hiding reversible en images binaires plaintext;
- methodes par patterns/blocs/mapping/covers binaires;
- travaux 2019-2026 ou travaux plus anciens decouverts par backward snowballing et structurants;
- travaux voisins sur cout auxiliaire, net payload, compression de location map, ou distortion-aware binary covers quand ils peuvent etre invoques par un reviewer.

Exclus:

- grayscale/color/JPEG/medical/DICOM si non specifique aux images binaires plaintext;
- encrypted-domain RDH quand le probleme scientifique est separable RDHEI, ciphertext binary images, secret sharing, MSB reservation ou encryption-then-compression;
- steganographie non reversible, watermarking robuste/fragile non RDH, coverless hiding;
- dual-image RDH et AMBTC/compressed-domain quand le support ou le protocole n'est pas une seule image binaire plaintext;
- surveys et papiers generaux sauf s'ils etablissent un concept transversal comme net capacity ou overhead/location map.

### Screening

Le screening a ete fait en trois niveaux:

1. titre: conserver uniquement RDH/data hiding binaire, block/pattern/mapping, ou 2025-2026 potentiellement proche;
2. resume/page editeur: exclure encrypted/JPEG/grayscale/dual-image/non-reversible;
3. article/page complete quand accessible: verifier side information, location map, block size, reversibilite, payload mesure, presence/absence de net accounting et steganalyse.

## Chaines de recherche documentees

### OpenAlex, 2019-01-01 a 2026-08-03

| Chaine | Resultats OpenAlex | Resultats pertinents apres screening |
|---|---:|---|
| `"reversible data hiding" "binary image"` | 272 | Ren 2019 encrypted binary, Zhang 2019 magnification, Huynh 2025, Li 2023, Xiao 2022 via snowballing |
| `"reversible data hiding" "binary images"` | 272 | meme ensemble; beaucoup de RDHEI faux positifs |
| `"reversible data hiding" "binary image" "block mapping"` | 4 | Huynh-Nguyen 2025 seul resultat directement proche |
| `"reversible data hiding" "binary image" "pattern substitution"` | 3 | Dong 2020, Li 2023, Ho 2009 par references |
| `"reversible data hiding" "binary image" "multi-embedding"` | 2 | Li et al. 2023 |
| `"reversible data hiding" "encrypted binary image"` | 24 | Ren 2019, Li 2020, Fan 2026; exclus comme encrypted-domain |
| `"reversible data hiding" "ciphertext binary image"` | 1 | Chen/Feng et al. 2025 multi-party ciphertext binary; exclu comme ciphertext-domain |
| `"binary image" "reversible data hiding" "2025"` | 22 | Huynh 2025; plusieurs RDHEI/general-image exclus |
| `"binary image" "reversible data hiding" "2026"` | 5 | Fan 2026 encrypted binary; APVCPC/general encrypted exclus |

### Crossref

| Chaine Crossref | Top resultats utiles |
|---|---|
| `reversible data hiding binary image block mapping` | Huynh-Nguyen 2025; Xuan/older histogram; binary-block encrypted; Liu 2025 block-size HS |
| `reversible data hiding binary image multi embedding` | Li et al. 2023; older binary covers; encrypted/general false positives |
| `reversible data hiding encrypted binary image` | Fan 2026; Li 2020; binary-block encrypted 2017 |

### Requetes web exact-title/DOI

- `"High-capacity reversible data hiding scheme based on block mapping for binary image"`
- `"High-Quality Reversible Data Hiding Based on Multi-Embedding for Binary Images"`
- `"General Distortion Based Reversible Data Hiding for Binary Covers"`
- `"Reversible Data Hiding for Binary Images Based on Adaptive Overlapping Pattern"`
- `"Reversible data hiding in binary images by flipping pattern pair with opposite center pixel"`
- `"Reversible Data Hiding Algorithm for Binary Image Based on Sliding Window"`
- `"Large Capacity Data Hiding in Binary Image black and white mixed regions"`
- `"Multi-Party Reversible Data Hiding in Ciphertext Binary Images Based on Visual Cryptography"`
- `"Reversible data hiding in encrypted binary image with high capacity"`
- `"The Relationship Between Block Size and Reversible Data Hiding Based on Image Block Histogram Shifting"`

## Articles inclus comme prior art direct ou proche

| Travail | Domaine | Apport etabli | Impact sur la nouveaute ABM |
|---|---|---|---|
| Ho et al. 2009, `10.1016/j.csi.2008.09.014` | RDH binaire, pattern substitution | Histograms de patterns, substitutions reversibles, meilleure capacite/qualite que PWLC | Etablit le principe pattern substitution; ne contient pas block mapping 3x3 net-aware ni codec enumeratif |
| Zhang et al. 2012, `10.1109/TIP.2012.2187667` | codes optimaux pour binary covers | Utilise une approche de codage pour approcher une borne rate-distortion et ameliorer des RDH sur sequences binaires | Important contre une revendication trop large sur "coding"; ne traite pas un wire codec PEAK/ZERO 3x3 ni benchmark net |
| Zhang et al. 2014, `10.1109/TIP.2014.2358881` | general distortion metrics | Probabilites de transition optimales pour metriques de distorsion generales | Montre que la distorsion generale existe; pas un ranker CNN distance-one pour blocks binaires 3x3 |
| Dong et al. 2015, `10.4218/etrij.15.0114.1058` | RDH binaire, overlapping PS | Generalise l'overlapping pattern substitution; class map, run lengths, meilleure capacite/PSNR | Explique la lignee Dong; pas de net accounting filaire comparable |
| Dong et al. 2020, `10.1186/s13635-020-00107-w` | RDH binaire, adaptive overlapping pattern | Choix contextuel PM/PF/PFR, location map, compression arithmetique, trois rounds | Contient side information et compression, mais pas selection par payload net exact ni codec enumeratif PEAK subset |
| Yin et al. 2020 PPOCP, `10.1016/j.jvcir.2020.102816` | RDH binaire, PPOCP | Paires 3x3 avec centre oppose, score distortion/payload, reversibilite | Plus conservateur et faible distortion; pas un block-mapping gross/net PEAK/ZERO |
| Yin et al. 2021 DESG, `10.1109/TCSVT.2020.3032685` | RDH halftone | Groupes d'etats dynamiques, overhead DESG, moins de flips | Adjacent binary/halftone; pas plaintext binary block mapping benchmark |
| Xiao et al. 2022, `10.1109/LSP.2022.3227813` | RDH pour binary covers | Distortion pixel-by-pixel, reconstruction information, matrix embedding | Tres important et absent du `.bib`; ne contient pas PEAK/ZERO block mapping, net wire codec ou CNN ranking |
| Ren et al. 2023, `10.1145/3573942.3574064` | RDH binaire, sliding window | Sliding window pour ameliorer capacite/qualite | Baseline/adjacent; pas net accounting filaire |
| Yang 2023, `10.1109/EIECT60552.2023.10441974` | data hiding binaire, mixed regions | Exploite regions noir/blanc mixtes, blocs 2x2 et tables d'encodage | Non explicitement RDH/wire-level dans les sources accessibles; ne bloque pas |
| Chhajed & Garg 2023, `10.12785/ijcds/130176` | data hiding binaire, BDPP | Blocks 3x3 partitionnes diagonalement, pixel central, faible distortion | Data hiding binaire proche mais pas RDH net-aware block mapping |
| Li et al. 2023, `10.3390/math11194111` | RDH binaire, multi-embedding | Distortion-based framework, cover decoupling, reconstruction information, matrix embedding; PSNR 49.45/SSIM 0.9705 a 1000 bits | Important et absent du `.bib`; ne contient pas block mapping PEAK/ZERO ni benchmark net/steganalysis |
| Huynh & Nguyen 2025, `10.1007/s11042-024-20553-9` | RDH binaire, block mapping 3x3 | Non-overlapping 3x3, histogramme de blocks non uniformes, highest-frequency block vers zero-frequency blocks, distance de Hamming, EC moyenne 2070 bits | Prior art le plus proche; impose de cadrer ABM comme net-capacity/wire-level + learned ranking + benchmark commun |

## Backward snowballing

### Depuis Huynh-Nguyen 2025

References pertinentes identifiees sur la page Springer:

- Zhang et al. 2019 image magnification, `10.1007/s11042-019-7519-2`;
- Dong et al. 2020 adaptive overlapping pattern, `10.1186/s13635-020-00107-w`;
- Yin et al. 2020 PPOCP, `10.1016/j.jvcir.2020.102816`;
- Ren et al. 2023 sliding window, `10.1145/3573942.3574064`;
- Yang 2023 mixed regions, `10.1109/EIECT60552.2023.10441974`;
- Chhajed & Garg 2023 BDPP, `10.12785/ijcds/130176`;
- Ho et al. 2009 pattern substitution, `10.1016/j.csi.2008.09.014`;
- Lu et al. 2004 DRD, `10.1109/LSP.2003.821748`.

Conclusion snowballing: Huynh-Nguyen couvre bien la lignee binary block/pattern; il ne cite pas Li 2023 multi-embedding ni Xiao 2022 comme elements centraux de general distortion binary covers.

### Depuis Dong 2020

References pertinentes identifiees sur la page Springer:

- Tseng/Pan 2000/2002 two-color image data hiding;
- Wu & Liu 2004 authentication/annotation;
- Yang & Kot 2007 connectivity-preserving pattern-based data hiding;
- Chiang et al. 2005 PWLC;
- Xuan et al. 2008 run-length histogram modification;
- Ho et al. 2009 pattern substitution;
- Dong/Kim 2011 efficient PS;
- Dong et al. 2015 overlapping PS.

Conclusion snowballing: Dong 2020 est une evolution directe de PS/overlapping PS avec location map et compression. C'est le meilleur precedent pour contester "side information ignored", mais son side information n'est pas transforme en critere de selection net-capacity comparable et executable entre methodes.

### Depuis Li et al. 2023

References pertinentes identifiees sur MDPI:

- Ho et al. 2009 pattern substitution;
- Zhang et al. 2012 optimal codes for binary covers;
- Zhang et al. 2014 optimal transition probability for general distortion metrics;
- Yin et al. 2020 PPOCP;
- Xiao et al. 2022 general distortion based RDH for binary covers.

Conclusion snowballing: Li/Xiao introduisent une ligne "distortion-based / matrix embedding / reconstruction information" pour binary covers. Cette ligne doit etre citee pour eviter une faille de litterature. Elle ne rend pas non nouveau le codec net-aware PEAK/ZERO ni le benchmark commun.

### Depuis PPOCP 2020

Forward/recommended ScienceDirect indique des citants ou articles proches:

- Li et al. 2023 multi-embedding;
- Chhajed & Garg 2023 BDPP;
- Yin et al. 2021 halftone DESG.

Conclusion snowballing: PPOCP reste une alternative faible distortion/detectabilite. Il ne couvre pas la maximisation de payload net apres serialization.

## Forward citation chasing 2025-2026

Source reproductible: OpenAlex `filter=cites:<work_id>,from_publication_date:2025-01-01,to_publication_date:2026-08-03`.

| Article source | Citants 2025-2026 trouves | Citants pertinents |
|---|---:|---|
| Huynh-Nguyen 2025 | 0 | Aucun citant indexe au 2026-08-03 |
| Yin PPOCP 2020 | 2 | Huynh-Nguyen 2025; Multi-Party RDH in Ciphertext Binary Images 2025 |
| Dong 2020 | 2 | Huynh-Nguyen 2025; un papier medical vision-transformer non binaire |
| Li et al. 2023 multi-embedding | 2 | Aucun block mapping binaire plaintext; citants general image/steganography |
| Ren sliding-window 2023 | 1 | Huynh-Nguyen 2025 |
| Yang 2023 mixed regions | 6 | Huynh-Nguyen 2025; autres faux positifs NLP/generation |
| Zhang magnification 2019 | 2 | Huynh-Nguyen 2025; Multi-Party ciphertext binary 2025 |
| Ren encrypted binary 2019 | 6 | Huynh-Nguyen 2025; Multi-Party ciphertext binary 2025; plusieurs RDHEI |
| Fan encrypted binary 2026 | 0 | Aucun |
| Xiao binary covers 2022 | 3 | General distortion JPEG 2025; ternary matrix JPEG 2026; general distortion metric MHM 2025 |
| Zhang optimal codes 2012 | 0 | Aucun |
| Zhang general distortion 2014 | 5 | RDHEI/general-image; aucun plaintext binary block mapping |
| Yin halftone DESG 2021 | 0 | Aucun |
| Ho pattern substitution 2009 | 2 | Multi-Party ciphertext binary 2025; encrypted 3D meshes 2026 |

Conclusion forward chasing: le seul citant 2025-2026 directement dans la zone plaintext binary block/pattern est Huynh-Nguyen 2025. Les autres citants recents sont encrypted-domain, JPEG/general-image, dual-image, medical ou non-binary.

## Table d'exclusion

| Travail | Decision | Justification |
|---|---|---|
| Ren et al. 2019, encrypted binary pixel prediction, `10.1016/j.sigpro.2019.07.020` | Exclu du prior art direct, cite adjacent | RDH en images binaires chiffrees; extraction/restauration en domaine chiffre/dechiffre; pas plaintext block mapping |
| Li et al. 2020, shared pixel prediction/halving compression, `10.1186/s13640-020-00522-6` | Exclu direct, cite adjacent | Encrypted binary image; probleme RDHEI different |
| Fan 2026, encrypted binary high capacity, `10.1007/s11042-026-21466-5` | Exclu direct, a signaler dans audit 2026 | Image binaire chiffree, room vacating par run-length coding; pas PEAK/ZERO plaintext |
| Chen/Feng et al. 2025, Multi-Party RDH in Ciphertext Binary Images, `10.1109/LSP.2025.3557273` | Exclu direct, cite adjacent 2025 | Ciphertext binary images + visual cryptography + multiparty; pas single-cover plaintext block mapping |
| APVCPC 2026, `10.3390/s26051636` | Exclu | RDH in encrypted images; general encrypted images, pas binaire plaintext |
| Unified RDH framework for block-scrambling EtC 2026, `10.3390/info17020118` | Exclu | Encryption-then-compression systems; pas binary image RDH |
| General distortion JPEG images 2025, `10.1007/s44443-025-00199-9` | Exclu direct, cite methode generale si besoin | JPEG RDH, pas image binaire; confirme seulement la ligne distortion-aware |
| High-Capacity RDH for JPEG using ternary matrix embedding 2026, `10.1109/TMM.2026.3651010` | Exclu | JPEG, ternary matrix embedding, pas binary block mapping |
| MME-based piecewise data transformation and 2D mapping 2026, `10.1109/TMM.2026.3664910` | Exclu direct | PEE/general RDH; "binary matrix embedding" concerne la transformation de donnees, pas des images binaires |
| Secure reversible image hiding with dual stego matrix encoding 2026, `10.1016/j.csi.2026.104131` | Exclu | Dual-image steganographic framework, AES/FTT/BLTM/LSB; pas single binary cover ni PEAK/ZERO |
| Dual-image Sudoku block mapping 2025, `10.70003/160792642025032602001` | Exclu direct | Dual-image RDH, Sudoku mapping, grayscale/dual cover protocol; pas image binaire plaintext unique |
| A high-payload data hiding method with voting/dynamic mapping 2025, `10.3390/electronics14173498` | Exclu | General images/pixel prediction; pas binary RDH block mapping |
| Relationship Between Block Size and RDH Based on Image Block Histogram Shifting 2025, `10.1109/ICICML67980.2025.11333645` | Exclu | Histogram shifting sur images standard; block size/HS general, pas binaire plaintext PEAK/ZERO |
| High-capacity RDH in encrypted images based on multi-predictions and efficient PBTL 2025, `10.1016/j.dsp.2025.105096` | Exclu | RDHEI/general images; "binary tree labeling" n'est pas binary-image block mapping |
| Reversible data hiding in encrypted DICOM images with fixed/block-wise pixel prediction 2025, `10.1016/j.sigpro.2025.110287` | Exclu | DICOM/encrypted medical, pas image binaire plaintext |
| SS-RDHEI with PDPM and auxiliary data free coding 2025, `10.1016/j.eswa.2025.129766` | Exclu | RDHEI, pas binary plaintext; utile seulement pour montrer que l'auxiliary coding existe ailleurs |
| Reversible Data Hiding in Halftone Images Based on DESG 2021, `10.1109/TCSVT.2020.3032685` | Exclu direct, cite adjacent | Halftone/binary-like, overhead DESG; pas 3x3 binary document block mapping |
| Chhajed/Garg 2023 BDPP, `10.12785/ijcds/130176` | Inclu adjacent, pas baseline direct | Data hiding binaire 3x3 diagonal; pas clairement RDH wire-level/net/steganalysis |
| Yang 2023 mixed regions, `10.1109/EIECT60552.2023.10441974` | Inclu adjacent, pas baseline direct | Data hiding binaire par blocks 2x2/mixed regions; pas net accounting ni codec |

## Verification des papiers 2025-2026 absents du `.bib`

### Presents dans `.bib`

- Huynh & Nguyen 2025, `10.1007/s11042-024-20553-9`: present.
- Ekodeck et al. 2026 non lie au RDH binaire: present mais hors audit scientifique.

### Absents mais a mentionner dans la discussion/audit

- Fan 2026 encrypted binary, `10.1007/s11042-026-21466-5`: a exclure explicitement comme encrypted-domain.
- Chen/Feng et al. 2025 Multi-Party RDH in Ciphertext Binary Images, `10.1109/LSP.2025.3557273`: a exclure explicitement comme ciphertext/multiparty.
- General distortion JPEG 2025, `10.1007/s44443-025-00199-9`: pas necessaire dans l'etat de l'art binaire, mais utile si le manuscrit discute distortion-aware RDH general.
- MME/2D mapping 2026, `10.1109/TMM.2026.3664910`: pas binaire, pas necessaire sauf pour eviter une confusion sur "2D mapping".
- Dual-image Sudoku block mapping 2025, DOI local `10.70003/160792642025032602001`: pas single-cover binary RDH.
- Relationship Between Block Size and RDH Based on Image Block Histogram Shifting 2025, `10.1109/ICICML67980.2025.11333645`: pas binaire.
- Secure reversible image hiding with dual stego matrix encoding 2026, `10.1016/j.csi.2026.104131`: dual-image/general steganography.

### Absents mais importants meme si 2022-2023

- Xiao, Li & Zhao 2022, `10.1109/LSP.2022.3227813`: important, car RDH pour binary covers avec modele de distorsion general et reconstruction information.
- Li et al. 2023, `10.3390/math11194111`: important, car RDH binaire multi-embedding, matrix embedding, reconstruction information, forte qualite visuelle.
- Yin et al. 2021 halftone DESG, `10.1109/TCSVT.2020.3032685`: adjacent, utile si reviewer evoque halftone/binary-like RDH.
- Zhang et al. 2012 optimal codes for binary covers, `10.1109/TIP.2012.2187667`: utile pour ne pas sur-revendiquer la partie "coding".
- Zhang et al. 2014 general distortion metrics, `10.1109/TIP.2014.2358881`: utile pour ne pas sur-revendiquer la partie "distortion metric".
- Ho et al. 2009 pattern substitution, `10.1016/j.csi.2008.09.014`: source fondamentale, actuellement seulement indirecte via Dong/Huynh.

## Corrections bibliographiques locales a faire

1. `Zhang2022Magnification` est probablement faux dans `paper/references.bib`: le travail est publie dans `Multimedia Tools and Applications`, volume 78, pages 21891-21915, en 2019, DOI `10.1007/s11042-019-7519-2`. Le manuscrit l'appelle encore Zhang 2022/JVCIR 78.
2. `Ren2019Encrypted` a un DOI local suspect: `10.1016/j.sigpro.2019.07.011` dans le `.bib`; les sources indexees donnent `10.1016/j.sigpro.2019.07.020`.
3. Le `.bib` manque les deux papiers binaires importants Xiao 2022 et Li 2023. S'ils ne sont pas cites, le reviewer peut dire que la revue ignore la ligne "general distortion / matrix embedding / reconstruction information".
4. Si la discussion veut soutenir que "net capacity" n'est pas revendiquee trop largement, ajouter un papier RDHEI/general qui definit total/net capacity ou soustrait labels/extra bits peut prevenir une objection.

## Position finale

Le rejet "lack of novelty" est levable si la nouveaute est formulee de maniere restrictive et technique:

> 3x3 block mapping, pattern substitution, location maps, matrix embedding, distortion-aware binary covers, and gross-capacity binary RDH are prior art. The technical contribution defended here is serialization-aware active mapping selection for plaintext binary-image PEAK/ZERO block mapping: active mapping tables are selected by useful payload after exact serialized auxiliary cost, the auxiliary stream is encoded by an enumerative codec, distance-one ZERO choices are ordered by a validated CNN, and the resulting method is compared against PPOCP, Dong, and Huynh-Nguyen under an executable common protocol with exact reversibility, paired statistics, resources, and grouped steganalysis.

Cette position est plus solide que "insufficient novelty" parce qu'elle ne contredit pas Huynh-Nguyen 2025: elle l'absorbe comme baseline le plus proche et deplace la nouveaute vers la comptabilite filaire, le codec, l'optimisation nette, l'apprentissage et le protocole experimental commun.

## Formulation courte pour reponse au reviewer

We agree that binary-image RDH, pattern substitution, 3x3 pattern pairs, distortion-aware binary covers, and gross-capacity block mapping are established. We therefore clarified that the contribution is not the existence of block mapping itself. Among the prior work identified through the systematic search and citation chasing, we found no plaintext binary-image PEAK/ZERO block-mapping method that uses the exact serialized description cost inside the active mapping-selection decision and implements that decision with an executable enumerative wire representation. Huynh-Nguyen 2025 remains the closest gross-capacity block-mapping prior art, while Xiao 2022 and Li 2023 establish the distortion/matrix-embedding binary-cover line.
