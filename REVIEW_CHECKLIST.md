# Checklist di revisione — top tagging e anomaly detection

Stato al 2026-11-12, prima della generazione da ~500 narrative per caso d'uso.
Ogni voce è una decisione di design già presa nel codice: va confermata o
corretta. Accanto, dove sta e cosa guardare. Le voci marcate **(AD)** valgono
solo per l'anomaly detection, **(TOP)** solo per il top tagging.

---

## 0. A monte: teacher, detector, surrogato

- [ ] **(TOP) Teacher ParT**: fine-tune da `ParT_kin.pt` (JetClass), seed 42/43/44,
      acc 0.9435 ± 0.0002, AUC 0.9876. Le narrative usano il seed 42.
      → `particle_transformer/`, `scripts/infer_TopLandscape_logits.sh`
- [ ] **(TOP) Target del surrogato**: logit difference `t = z_top − z_qcd` (logit
      grezzi, non softmax). → `predict_TopLandscape_logits.py`
- [ ] **10 osservabili** pre-registrate (m, m_SD, tau21, tau32, C2, D2, C3, N3, pT,
      n_const) con i loro parametri (beta = 1, z_cut = 0.1, troncamenti C3 a 50 e
      N3 a 40 costituenti). → `surrogate/basis_definitions.md`
- [ ] **EBM a zero interazioni** come surrogato; Ridge come limite inferiore, GBDT
      come limite superiore. → `surrogate/train_surrogates_top.py`
- [ ] **(AD) Detector**: IAD sul latente congelato `x_cls`, f = 10 %, seed 0. È il
      detector che spieghiamo. → `anomaly/transfer.py`,
      `narrative_anomaly.yaml: inputs`
- [ ] **(AD) Split del surrogato**: fit su metà di val, selezione sull'altra metà
      (per parità del jet), report su test; train escluso perché in-sample per
      l'IAD. → `anomaly/prepare_surrogate_inputs.py`
- [ ] **(AD) Fedeltà accettata**: EBM R² 0.761, ADO 0.948, accordo 0.898 a
      quantili allineati (a f = 10 %). È sufficiente per narrare?

## 1. Gate di astensione

- [ ] **(TOP)** Residuo `|r| = |t_clip − g_hat|`: full se < 1.0, caveat se < 2.0,
      abstain altrimenti (69.5 % full, 7.1 % abstain), calibrato sul test set.
      → `narrative_top.yaml: abstention`, `narrative/artefact.py`
- [ ] **(AD)** Sulla rarità (background efficiency) invece che sul logit; il
      riferimento sono i jet QCD della metà di selezione di val.
      → `anomaly/gate.py`
- [ ] **(AD)** Abstain se detector ed EBM non concordano sulla decisione al 10 %,
      in entrambe le direzioni.
- [ ] **(AD)** Caveat se entrambi segnalano il jet ma le rarità differiscono più di
      ×10 (fattore calibrato per avere ≤ 30 % di caveat tra i jet segnalati).
- [ ] **(AD)** Il caveat cade se le bande coincidono, il che può succedere solo dentro
      `extreme` (decisione del 2026-11-12). Il verdetto originale è in
      `meta.gate_regime`. → `artefact_anomaly.build`
- [ ] **(AD)** Sui jet che nessuno dei due segnala, la rarità non viene controllata.

## 2. Artefatto (ciò che il narratore vede)

- [ ] Ogni numero è arrotondato prima di essere passato (2 decimali per value, phi
      e pct), così il verifier confronta esattamente ciò che è stato inviato.
- [ ] Ciò che il modello non deve affermare è **assente** dall'artefatto, non
      vietato dal prompt.
- [ ] `top_k: null`: tutte e 10 le osservabili, ordinate per |phi|.
- [ ] pct e rarity sono limitati a 0.01 (mai 0 o 100, che si leggerebbero come
      "nessun jet più estremo").
- [ ] **(TOP)** phi orientati verso la decisione (positivo spinge verso
      `decision`); pct e rarity contro l'*altra* classe del tagger, mai contro
      l'etichetta vera. → `narrative/artefact.py`, `narrative/stats.py`
- [ ] **(TOP)** Sono presenti score_tagger, score_pct, score_reconstructed, recon_pct
      e intercept.
- [ ] **(AD)** Niente logit: decisione, rarità del detector come numero `[AR]` e come
      banda `[A]`, banda dell'EBM solo come parola `[AS]`.
      → `narrative/artefact_anomaly.py`
- [ ] **(AD)** Bande: extreme ≤ 0.1 %, strong ≤ 1 %, moderate ≤ 10 % (anomali),
      borderline ≤ 30 %, typical (background-like). Gli estremi superiori sono
      inclusi.
- [ ] **(AD)** phi **non** orientati: positivo vuol dire sempre "più anomalo". Le
      parole di direzione (`anomalous` / `background`) viaggiano nell'artefatto.
- [ ] **(AD)** pct e rarity delle osservabili sono contro il riferimento QCD, non
      contro una classe.
- [ ] **(AD)** L'etichetta vera non entra mai nell'artefatto.

## 3. Glossari

- [ ] **(TOP)** `glossary_top.yaml`: le 10 `meaning`, tutte `reviewed`. Sono
      corrette fisicamente e nel verso giusto (cosa indicano i valori bassi e cosa
      gli alti)?
- [ ] **(AD)** `glossary_anomaly.yaml`: m, tau32, D2 e pT sono riscritti senza nomi di
      particelle, le altre 6 sono uguali a top. Restano corrette senza dire "top"?
- [ ] Il glossario è l'**unica** licenza interpretativa: va bene che i
      troncamenti (C3 a 50, N3 a 40) e le correlazioni (C2 con m e pT) finiscano
      nelle narrative?

## 4. Prompt

- [ ] **(TOP)** `prompt_top.yaml`, id `top-v16`, refine `top-refine-v1`.
- [ ] **(AD)** `prompt_anomaly.yaml`, id `ad-v3`, refine `ad-refine-v1`.
- [ ] Struttura: BACKGROUND, (AD: DECISION AND BANDS), SIGN CONVENTION, INPUT,
      GLOSSARY, TASK, TAGGING, GUARDRAILS, DATA INPUT; più `remainder`, `refine` e
      `caveat`.
- [ ] Contratto dei tag: ogni numero dell'artefatto dentro un tag con l'indice a due
      cifre (`[V04]`, mai `[V4]`); nessuna cifra fuori dai tag.
- [ ] Guardrail comuni: riportare senza calcolare (niente somme, rapporti o
      confronti numerici); niente controfattuali; non correggere l'artefatto; non
      dire se il modello ha ragione; nessuna fisica oltre il glossario.
- [ ] Guardrail su pct e rarity: rarity misura quanto un valore è insolito, pct
      solo la direzione; pct alto vuol dire *pochi* jet sopra il valore.
- [ ] **(AD)** "Leggi il glossario nel verso del valore" (ad-v2): un C2 alto non va
      descritto con la lettura dei valori bassi.
- [ ] **(AD)** Divieto di identità: nessuna particella, processo o "signal". Il prompt
      vieta "top" **anche** nel senso comune, mentre il verifier ora lo accetta come
      rango o posizione ("top five"). Va bene che il prompt sia più severo del
      verifier?
- [ ] **(AD)** La decisione appartiene al detector, i contributi all'EBM: mai "il
      detector ha guardato X".
- [ ] **(AD)** Mai dare all'EBM una percentuale né confrontare le due rarità come
      numeri, solo "stessa banda" o "banda diversa".
- [ ] Testo del caveat: aprire dicendo che le osservabili non spiegano del tutto la
      decisione, e restare brevi.

## 5. Verifier (cosa conta come errore)

- [ ] Famiglie: Faith (`wrong_value`), Comp (`missing_tag`), Form
      (`malformed_tag`, `unknown_tag`), NoBare (`untagged_number`), Read
      (`wrong_reading`), più **(AD)** Ident (`identity_claim`). TOTAL è la loro
      congiunzione stretta. → `narrative/verifier.py`
- [ ] Numeri confrontati come valori, non come stringhe ("38" equivale a 38.0, 2.4
      non equivale a 2.37).
- [ ] Le parole sono obbligatorie solo per le prime 5 osservabili
      (`READING_TOP_K = 5`), ma vengono controllate ovunque compaiano.
- [ ] Bande di rarity (`[Q]`): extreme ≤ 2, unusual ≤ 10, ordinary oltre. Sono
      diverse dalle bande AD del detector: va bene avere due "extreme" con soglie
      diverse?
- [ ] Bande di posizione (`[S]`): low ≤ 25, mid ≤ 75, high oltre.
- [ ] Direzione (`[D]`): il segno di phi (≥ 0 conta come positivo).
- [ ] **(AD)** `[DEC]` accetta `background-like` (forma canonica) e `background`
      (alias per le narrative ad-v2).
- [ ] **(AD)** `banned_terms` in `narrative_anomaly.yaml`: "top" è vietato tranne come
      rango o posizione, "signal" solo come sostantivo. Ci sono altri termini da
      aggiungere (per esempio "particle", "resonance", "new physics")? Su ad100 non
      compaiono come affermazioni di identità.
- [ ] Nomi delle osservabili con cifre (C2, tau32, N3_b1) esclusi dal controllo
      NoBare, anche se scritti in modo diverso (`tau3/2`).
- [ ] Limite noto: il verifier **non** controlla la lettura fisica nella prosa. Sono
      i 2 errori di direzione trovati su ad100 (`TODO.md`); se ne occuperà la
      valutazione.

## 6. Feedback e refine loop

- [ ] Ogni violazione diventa una riga di correzione (FAITHFULNESS, COMPLETENESS,
      FORMAT, IDENTITY) più la storia de-duplicata dei tentativi precedenti, sul
      modello PONTE. → `narrative/feedback.py`
- [ ] `retry_mode: refine`, `refine_temperature: 0.2`, `max_attempts: 5`; viene
      consegnato il primo tentativo che passa, nulla se nessuno passa.
      → `narrative_*.yaml: llm`
- [ ] Generazione a `temperature 0`, `seed 42`, `thinking_budget 0` (gpt-oss con
      `reasoning_effort: low`, qwen con thinking spento).
- [ ] Nessuno stop anticipato (decisione del 2026-11-12). In `blocks_report --attempt
      final` è registrata la media dei tentativi dei jet consegnati entro 5
      (sezione ATTEMPTS TO DELIVERY).
- [ ] Ogni tentativo viene salvato, anche quelli falliti.

## 7. Campioni e report

- [ ] **(TOP)** I 100 jet di v17 sono fissati in un file e uguali per tutti i
      modelli.
- [ ] **(AD)** `select_anomaly_jets.py`: stratificato per banda del detector e,
      dentro ogni banda, metà top veri e metà QCD veri (23/23/24/20/10). È il
      disegno giusto anche per i ~500 jet?
- [ ] Generatori: gemini-3.5-flash, gemini-3.5-flash-lite, deepseek-flash,
      gpt-oss-120b (vLLM), qwen3.8-27b (vLLM). Restano questi cinque?
- [ ] `blocks_report` riverifica ogni narrativa con il verifier attuale (non si fida
      del verdetto registrato durante il run), con intervalli di Wilson e McNemar a
      coppie.
- [ ] Si riportano entrambi i numeri: `first` (modello da solo) e `final` (dopo il
      refine).

---

## Esito del controllo di Claude (2026-11-13)

> **Applicato il 2026-11-13**, con i report di v17 e ad100 rimasti identici:
> 1. nuovo gate di top (`narrative/gate_top.py`): abstain se il segno di ĝ è
>    diverso da quello di t; caveat se la differenza di rango supera X punti
>    percentili, con X = 7.5 calibrato su val (caveat ≤ 30 % dei jet tenuti).
>    Quote su test: 68 % full, 26 % caveat, 6 % abstain (prima 50 / 29 / 21);
> 2. campione di top stratificato: da fare insieme ai file per il run da 500;
> 3. `[AR]` 0.01 si scrive "0.01 percent or less" (prompt `ad-v4`);
> 4. niente più `-0.0` nell'artefatto di top: phi, score_reconstructed,
>    intercept e phi_other;
> 5. `W` confrontato rispettando le maiuscole;
> 6. regola sui bordi delle bande riformulata (`ad-v4`, `top-v17`).
> Resta da fare: il README (punto 7).

Verificato sul codice e sui file di output. Tutto il resto delle voci 0–7 torna.

**Da decidere**
1. **Il gate di top non è più calibrato.** Le soglie 1.0 e 2.0 (commento in
   `narrative_top.yaml`: 69.5 % full, 7.1 % abstain) vengono da un teacher
   precedente. Con il teacher seed 42, sull'intero test set: **50 % full, 29 %
   caveat, 21 % abstain**; su v17: 58 / 25 / 17. Anche il teacher unseeded dà
   52 / 29 / 19. Il README dice "the gate withholds the other ten". Va scelto se
   ricalibrare le soglie (per esempio su un'astensione obiettivo, come fatto per
   AD) o tenerle e correggere i numeri dichiarati. Il riferimento al
   `narrative/README.md` nel commento punta a un file rimosso.
2. **Disegno del campione di top.** È casuale (seed 2), mentre AD è
   stratificato. Con accuratezza 0.94 un campione casuale da 500 contiene circa
   30 errori del tagger. Per spiegare anche i casi sbagliati serve una
   stratificazione per decisione × correttezza, come in AD?

**Minori**
3. **(AD)** `[AR]` è limitato a 0.01: 4 jet su ad100 hanno una rarità reale
   inferiore (fino a 0.001 %) e la narrativa dice "0.01 percent". Si può dire
   nel prompt che 0.01 significa "0.01 o meno".
4. **(TOP)** L'artefatto di top non ha la correzione `-0.0` che ha quello AD:
   14 phi su 7000 in v17 valgono `-0.0`. Non ha causato errori (sono tutte
   osservabili di coda, dove `[D]` è opzionale), ma va allineato prima del run
   da 500.
5. Il termine vietato `W` (case-insensitive) colpisce anche una "w" isolata
   (per esempio "w.r.t."). Su ad100 non è mai comparso.
6. Il prompt dice che un valore esattamente su una soglia "takes the word above
   it", cioè la riga sopra nella tabella, che è la banda inferiore. È corretto
   ma ambiguo. Si potrebbe riformulare in `ad-v4` / `top-v17`, insieme al punto 3.
7. Il README riporta ancora i risultati di v16 (EBM R² 0.962, teacher non
   seedato); oggi il valore è 0.936.

**Verificati senza problemi**: il teacher in uso è il seed 42 (hash di ebm e
feature identici, statistiche di riferimento ricalcolate dopo il cambio);
EBM a zero interazioni; parametri delle osservabili coerenti con
`top_basis.yaml`; etichetta 1 = QCD nelle tabelle AD; riferimento del gate,
pct e rarity tutti sulla metà di selezione di val, che il surrogato non vede;
formule e direzioni dei glossari; soglie `[Q]` e `[S]` identiche tra prompt e
verifier; il prompt AD tratta già esplicitamente il doppio "extreme"; il
feedback per `[DEC]` dice ora "background-like".

---

### Esempi da leggere accanto alla checklist
- `narrative/outputs/anomaly/narratives/adpilot_prompt_example.txt`: un prompt AD
  completo così come lo vede il modello (scritto con ad-v1/v2).
- `ad100_report_{first,final}.txt`, `ad100_breakdown.txt`: risultati AD.
- `narrative/outputs/top/narratives/v17_report_{first,final}.txt`: risultati top.
