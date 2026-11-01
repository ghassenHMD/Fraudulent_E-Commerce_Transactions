# Détection de fraude e-commerce explicable

Application web interactive et API qui classent les transactions d'une plateforme e-commerce par niveau de risque, signalent les plus suspectes dans la limite d'un budget d'analyse, **choisissent le seuil d'alerte en minimisant un coût exprimé en montants**, et **expliquent chaque alerte** (SHAP, codes motifs).


> **Auteur :** _Ghassen Hamdi_ · **Application :** _[lien Streamlit](https://fraudulente-commercetransactions-vplikdenpfkqbphnbrphe8.streamlit.app/)_ · **API :** _[lien Render](https://fraudulent-e-commerce-transactions.onrender.com/)_ · **Dépôt :** https://github.com/ghassenHMD/Fraudulent_E-Commerce_Transactions

---

## 1. Problème et résultat en bref

Les analystes antifraude ne peuvent examiner qu'une fraction des transactions. L'outil doit donc **classer** (du plus au moins risqué), **décider** (quand alerter, selon le coût de la fraude manquée et celui de l'analyse) et **justifier** (pourquoi cette transaction).

| | Résultat (période de test, non utilisée pour construire le modèle) |
|---|---|
| Modèle retenu | LightGBM, poids de classe, arrêt précoce |
| PR-AUC · ROC-AUC | **0,331** · **0,764** (taux de fraude de référence : 5,0 %) |
| Seuil à coût minimal | 0,464, choisi sur la validation, confirmé sur le test |
| Au seuil retenu | 15,2 % de transactions alertées · précision 19,2 % · rappel 58,7 % · F2 0,416 · MCC 0,277 |
| Coût métier | **2,39 M** contre 8,07 M sans contrôle et 7,36 M en alertant tout (**−68 %** vs la meilleure alternative simple) |
| Fiabilité du seuil | coût au seuil de validation 2 386 332 contre 2 378 034 au meilleur seuil possible a posteriori (écart 0,35 %) |

---

## 2. Les données

*Fraudulent E-Commerce Transactions* (Kaggle, `shriyashjagtap/fraudulent-e-commerce-transactions`) : **1 472 952 transactions**, 16 colonnes, du 01/01/2024 au 03/04/2024. Taux de fraude : **5,01 %**.

| Constat | Conséquence |
|---|---|
| `customer_id` est unique sur chaque ligne (aucun client n'a deux transactions) | Les variables comportementales (vélocité, écart au profil) sont **impossibles** et n'ont pas été construites. Choix justifié par les données. |
| 521 âges ≤ 0 (minimum −16) | Imputés par la médiane du train (35). |
| `transaction_hour` cohérent avec la date dans 98 % des cas | L'heure est recalculée depuis la date. |
| Adresses identiques/différentes : 5,02 % / 4,98 % de fraude | Variable `addr_match` conservée, sans signal. |
| `ip_address`, adresses, `customer_location`, identifiants | Écartés (quasi uniques ou non exploitables bruts). |

---

## 3. Méthode

```
notebook.ipynb  ──►  artifacts.joblib  ──►  fraud_core.py  ──►  app.py   (Streamlit)
(tout le travail)    (modèle, seuil,        (variables, score,  api.py   (FastAPI)
                      scores, agrégats)      SHAP, coût)
```

**Variables (22).** Heure en codage cyclique (sin/cos), jour de la semaine, week-end, nuit, log du montant, quantité, âge, ancienneté du compte, correspondance des adresses, modalités encodées du moyen de paiement, de la catégorie de produit et de l'appareil.

**Séparation chronologique 60 / 20 / 20**, sans mélange (la fraude doit être prédite à partir du passé) :

| Bloc | Lignes | Période | Taux de fraude |
|---|---|---|---|
| Entraînement | 883 771 | 01/01 → 26/02 | 5,02 % |
| Validation | 294 590 | 26/02 → 15/03 | 5,03 % |
| Test | 294 591 | 15/03 → 03/04 | 4,99 % |

Le taux de fraude est stable : pas de dérive entre les blocs.

**Déséquilibre.** Quatre stratégies comparées en validation avec le même modèle, appliquées au seul jeu d'entraînement :

| Stratégie | PR-AUC val | ROC-AUC val |
|---|---|---|
| Aucune | 0,3331 | 0,7688 |
| **Poids de classe (retenue)** | 0,3337 | 0,7667 |
| Sous-échantillonnage 1:5 | 0,3348 | 0,7685 |
| SMOTE 1:5 | 0,3352 | 0,7684 |

Les écarts (< 0,003) sont de l'ordre du bruit. Le poids de classe a été retenu pour sa simplicité : il n'invente aucune donnée et ne coûte rien en temps.

**Modèles comparés** (4 méthodes) :
- *supervisés* : régression logistique pondérée (référence interprétable) et **LightGBM** ;
- *non supervisés*, sans aucune étiquette à l'entraînement : **Isolation Forest** et **LOF** (mode *novelty*, ajusté sur un sous-échantillon de 30 000 lignes).

**Coût métier.** `coût = montants des fraudes manquées + coût d'analyse × nombre d'alertes`. Hypothèses : **25 par alerte** (≈ 20 minutes d'analyste plus gêne client) et **capacité de 300 dossiers par jour**, soit un budget d'alertes `k` = 5 700 sur la période (≈ 1,9 % du volume). Le seuil est choisi sur la validation en parcourant toutes les coupures possibles, puis appliqué tel quel au test.

---

## 4. Résultats

### Comparaison des méthodes (test)

| Méthode | PR-AUC | ROC-AUC | Précision@k | Rappel@k | Coût (seuil de validation) |
|---|---|---|---|---|---|
| **LightGBM** | **0,331** | **0,764** | **53,6 %** | **20,8 %** | **2 386 332** |
| Régression logistique pondérée | 0,249 | 0,717 | 44,2 % | 17,2 % | 3 292 561 |
| LOF | 0,098 | 0,619 | 20,0 % | 7,8 % | 4 292 318 |
| Isolation Forest | 0,065 | 0,583 | 8,4 % | 3,3 % | 5 709 093 |

Référence aléatoire : PR-AUC et précision@k ≈ 0,050. Le modèle supervisé est nettement meilleur ; les méthodes non supervisées ne dépassent que de peu le hasard, ce qui indique que **les fraudes de ce jeu synthétique ne sont pas des points aberrants** dans l'espace des variables.

### Sensibilité au coût d'analyse (test)

| Coût par alerte | % alertées | Précision | Rappel | F2 | Coût | Sans modèle | Tout alerter |
|---|---|---|---|---|---|---|---|
| 5 | 41,7 % | 8,9 % | 74,3 % | 0,301 | 1 053 070 | 8 070 789 | 1 472 955 |
| 10 | 28,0 % | 12,0 % | 67,3 % | 0,349 | 1 519 959 | 8 070 789 | 2 945 910 |
| **25** | **15,2 %** | **19,2 %** | **58,7 %** | **0,416** | **2 386 332** | 8 070 789 | 7 364 775 |
| 50 | 4,4 % | 35,4 % | 31,1 % | 0,319 | 2 986 392 | 8 070 789 | 14 729 550 |
| 100 | 2,8 % | 43,7 % | 24,6 % | 0,270 | 3 477 457 | 8 070 789 | 29 459 100 |

Le seuil dépend fortement de l'hypothèse de coût, d'où la possibilité de la modifier dans l'application.

### Explicabilité (SHAP)

Deux variables dominent largement : l'**ancienneté du compte** (influence moyenne 0,40) et le **montant** (0,24). Toutes les autres pèsent moins de 0,012 (quantité, âge, heure, catégorie, moyen de paiement…). Une alerte typique se lit donc : « compte récent » et/ou « montant élevé ». Les codes motifs de l'application regroupent les variables techniques en critères métier (par exemple l'heure et son codage cyclique forment un seul critère « Heure »), ce qui conserve l'additivité de SHAP.

---

## 5. Points d'attention et limites

- **Données synthétiques.** Les performances en production réelle devront être revérifiées. Un signal porté par deux variables seulement peut refléter les règles de la simulation plutôt que des comportements réels.
- **Signal modéré** (PR-AUC 0,33). À ce niveau, aucun seuil ne combine précision et rappel élevés.
- **Seuil optimal et capacité de l'équipe.** À 25 par alerte, le seuil à coût minimal déclenche environ **2 360 alertes par jour**, soit près de 8 fois la capacité supposée (300). Traiter seulement 300 dossiers par jour donne une précision de 54 % mais un rappel de 21 %. C'est un arbitrage métier : renforcer l'équipe, automatiser un premier filtre, ou accepter un rappel plus faible. L'application met cet écart en évidence.
- **Hypothèses de coût** (25 par alerte, 300 dossiers par jour) : non issues de données réelles, elles sont paramétrables et testées par la table de sensibilité ci-dessus.
- **Équité.** Le modèle utilise `customer_age`. Son effet sur les alertes devrait être contrôlé (taux d'alerte par tranche d'âge) avant tout usage réel.
- **Un score n'est pas une probabilité** : la pondération des classes le déforme. Il sert à classer, et le seuil est choisi empiriquement sur le coût, sans hypothèse de calibration.
- **Arrêt précoce sur la validation** : la validation est légèrement optimiste, c'est pourquoi le test arbitre.
- Une alerte est une **aide à la décision**, pas un motif de blocage automatique.

---

## 6. Contenu du dépôt

```
Fraud_project/
├── notebook.ipynb         Tout le travail : données, variables, séparation, modèles,
│                          évaluation, seuil, SHAP, export de l'artefact
├── artifacts.joblib       Artefact complet pour l'application (16 Mo)
├── artifacts_api.joblib   Artefact allégé pour l'API (< 300 Ko)
├── fraud_core.py          Noyau commun : variables, score, explication, coût
├── app.py                 Application Streamlit
├── api.py                 API FastAPI
├── make_deploy.py         Génère l'artefact allégé et les dépendances épinglées
├── requirements.txt       Dépendances de l'application
├── requirements-api.txt   Dépendances de l'API
├── render.yaml            Déploiement de l'API sur Render
└── .python-version
```

`fraud_core.py` est la **source unique** de la préparation des variables : l'application et l'API l'importent toutes deux, ce qui exclut tout écart entre l'entraînement et la production. `python fraud_core.py` le vérifie : il reconstruit les variables et les scores d'un échantillon de 500 transactions et les compare à ceux du notebook (écart attendu : 0).

---

## 7. L'application (Streamlit)

Aucun jargon technique : l'interface suit le travail d'un analyste.

| Onglet | Contenu |
|---|---|
| **Alertes** | Choix d'une journée (période de test) ou import d'un CSV. File des alertes classées par risque, avec leurs motifs principaux, export CSV. Un clic sur une ligne affiche l'explication détaillée (graphique waterfall). |
| **Analyser une transaction** | Formulaire de saisie : niveau de risque, décision, motifs et waterfall. |
| **Performance et coûts** | Précision, rappel, F2, MCC, coût avec et sans modèle, courbes coût et précision-rappel. Volet repliable : comparaison avec les autres méthodes et courbes superposées. |
| **Tendances de fraude** | Taux de fraude dans le temps, par heure, catégorie, moyen de paiement, appareil, ancienneté du compte ; distribution des montants par classe. |
| **Comprendre le modèle** | Critères les plus influents et graphique SHAP (beeswarm). |

La barre latérale règle le déclenchement des alertes : **seuil recommandé** (coût minimal), **capacité de traitement** (les N dossiers les plus risqués par jour) ou **seuil manuel**, ainsi que le coût d'analyse et la capacité. Alertes, précision, rappel et coût se mettent à jour en direct, avec un avertissement lorsque la charge dépasse la capacité de l'équipe.

Format du CSV importé : `transaction_date`, `transaction_amount`, `payment_method`, `product_category`, `device_used`, `quantity`, `customer_age`, `account_age_days` ; optionnels : `transaction_id`, `addr_match` (ou `shipping_address` et `billing_address`). Les noms de colonnes d'origine du jeu Kaggle sont acceptés.

---

## 8. L'API (FastAPI)

| Route | Rôle |
|---|---|
| `GET /health` | État du service (public) |
| `GET /model` | Modèle, seuil, métriques de test, modalités acceptées |
| `POST /predict` | Score, niveau de risque, décision d'alerte |
| `POST /explain` | Idem, avec codes motifs et contribution de chaque critère |
| `POST /predict/batch` | Jusqu'à 10 000 transactions |

Documentation interactive : `/docs`. Le paramètre `?threshold=0.3` remplace le seuil à coût minimal. Une modalité inconnue renvoie une erreur 422 listant les valeurs acceptées. Si la variable d'environnement `API_KEY` est définie, les routes (hors `/health`) exigent l'en-tête `X-API-Key`.

```bash
curl -X POST http://127.0.0.1:8000/explain -H "Content-Type: application/json" -d '{
  "transaction_date": "2024-03-20T03:14:00", "transaction_amount": 830.5,
  "payment_method": "credit card", "product_category": "electronics", "device_used": "mobile",
  "quantity": 2, "customer_age": 29, "account_age_days": 9, "addr_match": false}'
```

Réponse (extrait) : `score`, `risk_level`, `alert`, `reason_codes` (ex. « Ancienneté du compte : 9 jours », « Montant : 830.50 »), `contributions`.

---

## 9. Installation et exécution

Les versions de `pandas`, `numpy`, `scikit-learn`, `lightgbm` et `joblib` doivent être **celles du notebook** (l'artefact est un objet picklé ; elles sont épinglées dans les fichiers `requirements`).

```bash
git clone https://github.com/ghassenHMD/Fraudulent_E-Commerce_Transactions.git
cd Fraudulent_E-Commerce_Transactions
python -m venv .venv && source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-api.txt                       # pour lancer aussi l'API
python fraud_core.py                                      # contrôle de parité avec le notebook
streamlit run app.py                                      # http://localhost:8501
uvicorn api:app --reload                                  # http://127.0.0.1:8000/docs
```

**Reproduire le travail.** Placer `Fraudulent_E-Commerce_Transaction_Data.csv` à côté du notebook, ajuster `PATH` dans la première cellule, exécuter toutes les cellules (la dernière écrit `artifacts.joblib`), puis `python make_deploy.py`.

---

## 10. Déploiement

| Composant | Plateforme | Configuration |
|---|---|---|
| Application | Streamlit Community Cloud | dépôt GitHub, branche `main`, fichier `app.py`, `requirements.txt` |
| API | Render (Blueprint) | `render.yaml` : `uvicorn api:app`, `ARTIFACTS_PATH=artifacts_api.joblib`, variable secrète `API_KEY` |

Une mise à jour du modèle suit toujours le même chemin : ré-exécuter la dernière cellule du notebook, remplacer `artifacts.joblib`, lancer `python make_deploy.py`, puis `git add -A && git commit && git push`. Les deux services se redéploient automatiquement.

Remarques sur les offres gratuites : l'API Render s'endort après 15 minutes sans trafic (réveil d'environ une minute), et l'application Streamlit s'endort après une période d'inactivité. Appeler `/health` avant une démonstration.

---

## 11. Données et licence

Jeu de données : *Fraudulent E-Commerce Transactions* (Kaggle, synthétique). Vérifier la licence sur la page du jeu avant toute réutilisation. Le jeu brut n'est pas versionné (`*.csv` est exclu par `.gitignore`).