# Revue de nouveaute: Net-Aware Learned Block Mapping for Binary-Image RDH

Date: 2026-08-03

## Position courte

La nouveaute defendable de l'article n'est pas "un nouveau block mapping 3x3" pris isolément. Ce terrain est deja occupe par les approches a motifs/blocs, en particulier Huynh et Nguyen (2025).

La position solide est la suivante:

> L'article introduit, dans le cadre du RDH en images binaires par block mapping, un protocole complet de capacite utile au niveau filaire: les bits applicatifs sont mesures apres serialization exacte de l'information auxiliaire, les paires PEAK/ZERO sont selectionnees par maximisation de capacite nette, le flux auxiliaire est code par un codec enumeratif compact, les ZERO distance-one sont ordonnes par un predicteur CNN valide hors image, et l'ensemble est compare a PPOCP, Dong et Huynh-Nguyen sous memes images, messages, charges, DRD, verification de reversibilite, cout auxiliaire, temps, memoire et steganalyse groupee.

Formulee ainsi, la contribution n'est pas une simple variation incrementale de Huynh-Nguyen. C'est une contribution methodologique et experimentale: passer de "gross embedding capacity" a "useful net payload under an executable reversible wire format".

## Ce qui existe deja

1. RDH binaire par motifs ou blocs.
   - Les methodes historiques utilisent PWLC, run-length histograms, pattern substitution, ou blocs 3x3.
   - Xuan et al. (ICPR 2008) utilisent la modification d'histogrammes de run-length et une carte de localisation pour la reversibilite.
   - Source: https://digitalcommons.njit.edu/fac_pubs/13010/

2. Pattern substitution adaptatif.
   - Dong et al. (2020) ameliorent le pattern substitution par selection contextuelle de PM/PF/PFR, resolution du miscoding en recouvrement, et compression/embedding multi-round de location map.
   - Leur apport est l'adaptation locale des motifs dans le domaine de difference, pas un block mapping PEAK/ZERO 3x3 avec comptabilite filaire nette.
   - Source: https://link.springer.com/article/10.1186/s13635-020-00107-w

3. PPOCP.
   - Yin et al. (2020) construisent des paires de motifs 3x3 avec pixels centraux opposes et choisissent une paire optimale via un score equilibrant distorsion et charge.
   - Leur objectif principal est la qualite visuelle et la reversibilite par PPOCP, pas la maximisation de payload net apres cout auxiliaire serialise.
   - Source: https://www.sciencedirect.com/science/article/abs/pii/S1047320320300663

4. Image magnification.
   - Zhang et al. agrandissent l'image et cachent les donnees dans l'image agrandie. C'est reversible mais change la taille du support; la capacite brute n'est donc pas directement comparable a protocole taille fixe.
   - Source: https://www.researchgate.net/publication/332220189_Reversible_data_hiding_in_binary_images_based_on_image_magnification

5. Block mapping 3x3 haute capacite.
   - Huynh et Nguyen (2025) divisent l'image binaire en blocs 3x3 non chevauchants, construisent un histogramme de motifs non uniformes, mappent un motif frequent vers des motifs de frequence zero et utilisent la distance de Hamming pour limiter la distorsion.
   - C'est le prior art le plus proche. Il etablit le block mapping 3x3 haute capacite, mais rapporte une capacite d'insertion brute; le cout exact du flux de synchronisation/metadata executable n'est pas comptabilise comme payload negatif.
   - DOI: https://doi.org/10.1007/s11042-024-20553-9

6. Sliding window et regions mixtes.
   - Ren et al. (ACM 2022/2023) proposent un RDH binaire par fenetre glissante pour ameliorer capacite et qualite visuelle.
   - Yang (IEEE 2023) cible les regions noir-blanc mixtes pour augmenter la capacite.
   - Ces travaux ne fournissent pas un benchmark commun avec cout auxiliaire filaire exact.
   - Sources: https://doi.org/10.1145/3573942.3574064 ; https://ieeexplore.ieee.org/document/10441974/

7. RDH en images binaires chiffrees.
   - Ren et al. (2019), Li et al. (2020) et Fan (2026) traitent un probleme different: cacher dans une image binaire chiffree ou separer extraction/dechiffrement/restauration.
   - Ces travaux contiennent parfois des sequences de localisation et compression auxiliaire, mais dans un modele encrypted-domain different du block mapping PEAK/ZERO en image binaire claire.
   - Sources: https://www.sciencedirect.com/science/article/abs/pii/S0165168419302737 ; https://link.springer.com/article/10.1186/s13640-020-00522-6 ; https://doi.org/10.1007/s11042-026-21466-5

## Ce qui est nouveau dans l'article soumis

1. Criterium de capacite nette au niveau filaire.
   - L'article ne compare pas seulement des bits inseres. Il definit:
     `C_net = C_gross - |A_wire|_2`
   - Le flux auxiliaire contient longueur, nombre de tables, rang du sous-ensemble PEAK actif et positions de flips.
   - Cette mesure change la question scientifique: "combien de bits utiles restent apres ce qu'il faut vraiment transmettre pour restaurer exactement ?"

2. Codec auxiliaire enumeratif pour block mapping binaire 3x3.
   - Le sous-ensemble actif de PEAKs est code par systeme combinatoire.
   - Les positions de flip distance-one sont codees en base 9.
   - Le flux utilise le plus court nombre d'octets capable de representer `binom(512,t) 9^t`.
   - A ma connaissance, aucun prior art identifie en RDH binaire plaintext par block mapping ne formule ce cout exact ni ne selectionne les mappings en fonction de ce cout.

3. Selection des mappings par maximisation du payload utile.
   - Les paires ne sont pas seulement choisies pour maximiser la capacite brute ou minimiser la distance.
   - Les PEAKs sont ordonnes, puis le prefixe actif est choisi pour maximiser `C_gross - |A_wire|_2`.
   - Si aucun sous-ensemble n'est net-positif, l'image reste inchangee. Cette decision est absente des methodes qui rapportent seulement EC/gross capacity.

4. Ranking appris des ZERO distance-one.
   - Le CNN classe les candidats ZERO a distance de Hamming 1 selon un cout perceptuel predit.
   - Validation leave-one-image-out sur 13 808 candidats: MAE 0.0367, correlation 0.9893.
   - Test operationnel separe: DRD moyen 0.8054 -> 0.7663, soit 4.85% de reduction, amelioration sur les 8 images.
   - Cela distingue l'article d'un simple seuil de Hamming fixe comme Huynh-Nguyen.

5. Protocole commun executable.
   - PPOCP, Dong, Huynh-Nguyen et ABM sont evalues avec memes images, memes messages, memes cibles 64/128/256/512 bits, meme DRD, memes checks de reversibilite, et cout auxiliaire explicite.
   - Cette comparaison est plus forte qu'une table de valeurs publiees non comparables.
   - Resultat cle: a 256 bits, ABM atteint 93 bits nets moyens sur 69 images matched, contre -81 pour Dong, -1812 pour Huynh-Nguyen, -2856 pour PPOCP; avantage net pairwise significatif apres Holm.

6. Extension de validation hors corpus avec BOSSbase.
   - Le ranker entraine sur documents est transfere sans retrain sur 100 images BOSSbase binarisees disjointes.
   - Resultat regenere: 669.6 bits bruts, 377.1 bits auxiliaires, 292.5 bits nets moyens; 75/100 images net-positives; restauration exacte.

7. Evaluation de steganalyse groupee.
   - Les papiers RDH binaires comparables se limitent en general a capacite, PSNR/DRD et reversibilite.
   - Ici, les courbes AUC logistique et Random Forest sont mesurees avec GroupKFold pour maintenir cover/stego dans le meme fold.
   - L'article ne cache pas la limite: ABM maximise le net payload, tandis que PPOCP reste meilleur en faible detectabilite.

## Ligne de demarcation avec Huynh-Nguyen 2025

Huynh-Nguyen est le concurrent a traiter frontalement. La demarcation doit etre explicite:

- Meme famille generale: block mapping binaire 3x3, PEAK/ZERO, distance de Hamming.
- Difference 1: Huynh-Nguyen optimise et rapporte une capacite brute; l'article soumis optimise et rapporte une capacite utile apres stream auxiliaire executable.
- Difference 2: Huynh-Nguyen utilise une selection fondee sur distance/Hamming; l'article soumis apprend l'ordre des ZERO distance-one par CNN et valide l'impact DRD hors image.
- Difference 3: Huynh-Nguyen ne fournit pas un cout serialise de synchronisation comparable; l'article soumis encode ce cout et l'integre dans la decision d'utiliser ou non chaque table.
- Difference 4: l'article soumis compare aussi PPOCP, Dong et Huynh-Nguyen dans un protocole commun avec tests statistiques, ressources et steganalyse.

Conclusion: si le reviewer lit "nouveau block mapping", il peut rejeter. S'il lit "serialization-aware mapping selection with an executable enumerative wire representation", le rejet pour manque de nouveaute devient beaucoup plus difficile.

## Formulation recommandee pour le manuscrit

Phrase centrale a placer dans l'introduction et la reponse aux reviewers:

> The novelty is not the use of 3x3 block mapping alone. The technical contribution is serialization-aware active mapping selection for binary-image PEAK/ZERO block mapping: the method selects active PEAK/ZERO mappings by useful payload after charging an explicitly serialized auxiliary stream, encodes that stream with an enumerative codec, learns distance-one candidate ordering from held-out image evidence, and evaluates ABM, PPOCP, Dong, and Huynh-Nguyen under a common reversible protocol with paired statistics, resource measurements, and grouped steganalysis.

## Formulation recommandee pour une reponse a "lack of novelty"

> We agree that 3x3 binary block mapping itself is established, most recently by Huynh and Nguyen (2025). We have therefore clarified that our contribution is not the existence of block mapping, but its conversion into an executable net-capacity framework. Prior binary-image RDH studies generally report gross embedding capacity, while the side information required for exact recovery is either not serialized, not charged, or not comparable across methods. Our manuscript defines `C_net = C_gross - |A_wire|_2`, designs an enumerative auxiliary codec for the active PEAK subset and flip positions, selects mapping prefixes by net payload, validates CNN-ranked distance-one assignment on held-out images, and compares ABM, PPOCP, Dong, and Huynh-Nguyen under identical covers, messages, payload targets, distortion computation, reversibility checks, auxiliary-bit accounting, runtime/memory profiling, and grouped steganalysis. Under this protocol, ABM is not claimed to dominate all objectives; it establishes the highest observed useful net-payload trajectory, while PPOCP remains the lowest-distortion and least-detectable alternative. This precise, wire-level and statistically paired positioning is absent from the prior literature and constitutes the novelty of the work.

## Ce qu'il ne faut pas sur-revendiquer

- Ne pas dire que l'article invente le RDH binaire.
- Ne pas dire qu'il invente le block mapping 3x3.
- Ne pas dire qu'il domine PPOCP en qualite visuelle ou securite: les resultats montrent l'inverse pour certains objectifs.
- Ne pas dire que BOSSbase est nouveau en RDH au sens large: il est deja utilise dans le RDH chiffre. Dire plutot que l'article l'utilise comme test externe disjoint dans un protocole plaintext block-mapping avec payload net.
- Ne pas dire que le CNN rend la methode securisee: la steganalyse montre un footprint croissant.

## Verdict

La nouveaute est suffisante si elle est cadrée comme une contribution de protocole, de comptabilite et de codec, avec un module appris valide experimentalement. Elle est faible si elle est presentee comme une simple amelioration de capacite d'un block mapping 3x3. Le manuscrit doit donc mettre en premiere ligne:

1. net-capacity wire-level accounting;
2. enumerative synchronization codec;
3. learned distance-one candidate ordering;
4. common executable benchmark with exact reversibility, paired statistics, resources, and steganalysis.

Cette combinaison precise n'apparait pas dans les travaux identifies.
