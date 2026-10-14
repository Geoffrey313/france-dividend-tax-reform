# Session du 13 octobre 2026. Extension de recherche, asymétrie fiscale, robustesses

Note de travail interne, hors dépôt publié (docs/internal est gitignored). Elle récapitule ce qui a été produit, décidé, et ce qui reste ouvert. Les chiffres renvoient aux pages de résultats Outline et aux scripts.

## Point de départ

Le cœur empirique de Phase 1 était déjà établi : double différence firme France contre cinq voisins (Allemagne, Italie, Espagne, Belgique, Pays-Bas), sur Orbis, la flat tax de 2018 ne faisant entrer aucun capital neuf (H1 nulle) et ne soutenant pas la thèse de Jalal (H2 non soutenue), avec un bond agrégé de 32 % des dividendes des ménages. La session a étendu l'analyse, ajouté l'univers coté, construit l'asymétrie 2013 contre 2018, et verrouillé les confondants.

## Ce qui a été produit

### Extensions au niveau firme et agrégé
- Phase 4, hétérogénéité par actionnariat personne physique et par taille. Même sur les sociétés détenues par une personne physique, pas d'entrée au capital. Extraction du drapeau d'actionnariat six pays (data/raw/ownership_flags.parquet, France 80,4 %). Les triples trésorerie et réserves sont des niveaux poolés confondus par le COVID, décomposés par année.
- Phase 5, français ou mondial. Dividendes des ménages indexés en base 2017, six pays plus États-Unis et Japon (OCDE). La France rompt seule en 2018, distinctive parmi les comparables. Élargi à 27 pays, la France n'est plus isolée, ce qui est écrit comme nuance et non comme pic unique.
- Phase 6b, destination. Le surcroît de 2018 se loge en dépôts et actions de portefeuille de marché secondaire, pas en capital de sociétés fermées ni en immobilier.
- Phase 6a, source. Ni les réserves ni la trésorerie des sociétés ne baissent, la hausse de trésorerie étant un fait du COVID. La source directe du payout n'est pas mesurable, Orbis ne portant pas les dividendes, donc preuve indirecte.
- Univers coté, Compustat Global 2010-2022. Les cotées ne lèvent pas de capital neuf, servent de contrôle négatif et de contrepartie publique au test privé. Batterie complète et hétérogénéité par taille.

### L'apport, l'asymétrie 2013 contre 2018, le cliquet
- SQ1, distribution. Semi-élasticité des dividendes des ménages, choc 2013 moins 0,40, choc 2018 plus 0,18 sur base lissée, la baisse ne défaisant que 45 % de l'effet de la hausse, 76 % sur le flux brut. Série Eurostat D42 étendue à 2010.
- SQ2, capital. La hausse de 2013 réalloue vers les bénéfices retenus, rétention plus 0,094 et fonds propres plus 0,109, la baisse de 2018 beaucoup moins, plus 0,033 et plus 0,040, soit une réponse environ deux fois et demie à trois fois plus faible. Panel Orbis étendu à 2010.
- Contrôle positif Boissel et Matray, la réallocation de 2013 est retrouvée. Contrôle négatif coté, rien là où le canal ne joue pas.

### Mécanisme, rattrapage avorté de 2013
Lecture possible du null de 2018 comme plancher déjà atteint. Étude d'événement du niveau de capital retenu, référence 2012. Statut explicite d'hypothèse corroborée et non établie. L'épreuve pays par pays montre que la France n'est pas l'exception nette, l'Espagne sautant plus fort, mais l'Espagne est un traité caché, voir ci-dessous. Confondant de crise non levé, inférence mince, plateau qui s'érode après 2020.

### Robustesses verrouillées
- Confondant des allocations pour fonds propres de type ACE. Vérifié et sourcé que la Belgique (intérêts notionnels, 2006), l'Italie (aide à la croissance économique, 2011-2018 puis 2020) et l'Espagne (réserve de capitalisation, 2015) subventionnent les fonds propres, soit la marge de SQ2. L'Espagne est donc un traité caché. Avec un indicateur d'allocation par pays et par année fixé sur la loi, sans retirer de pays, le cliquet de capital survit, rétention 2013 passant de plus 0,094 à plus 0,106, l'asymétrie passant d'environ deux fois et demie à environ trois fois et demie. Contre les seuls contrôles exempts, l'Allemagne et les Pays-Bas, le motif est encore plus asymétrique.
- Confondant de cycle. Avec un contrôle de chômage par pays et par année, le coefficient de 2013 reste quasi inchangé, le resserrement de l'asymétrie à environ deux fois venant du dénominateur, c'est 2018 qui se relève. Argument décisif, le cycle ne peut pas fabriquer l'asymétrie, 2018 n'étant pas une année de crise, l'objet du papier étant le contraste 2013 contre 2018, immunisé par construction contre le cycle commun.

## Décisions prises
- Pool d'identification, cinq voisins comparables. L'élargissement à des économies dissemblables dilue par construction, il ne réfute pas, il conforte que l'effet est propre aux comparables.
- Inférence par triangulation, contrôles positif et négatif plus cohérence SQ1 et SQ2, non par une valeur p de permutation, mécaniquement faible à six pays.
- Callaway et Sant'Anna. Implémenté pour de bon sur les hausses d'impôt multi-pays, avec panel de taux OCDE et covariable chômage. Résultat inconcluant, ATT global plus 0,66 non significatif, car les hausses sont hétérogènes en ampleur et l'agrégat ménages est bruité. Décision, le CS ne figure pas dans le papier, on reste sur le 2×2. Ne jamais écrire que le résultat tient sous CS.
- Mécanisme, au statut hypothèse corroborée non établie, jamais dans le résumé.

## Manuscrit
Brouillon complet écrit dans la page Outline Paper, structure SSRN, forme paper-writing-rules, un diagramme Mermaid cartographiant les tests et les hypothèses au début du modèle empirique, et les robustesses ACE et cycle ajoutées au paragraphe six. Références nettoyées, Sun et Abraham retiré car estimateur non utilisé. À vérifier par l'auteur, l'email de contact et la date de Boissel et Matray.

## Pointeurs Outline
- Results racine eFs8uF6kUM, synthèse en sept points et carte des phases.
- Paper yTkx3zuTwi, manuscrit et diagramme.
- Spec asymétrie 345e3e4b, SQ1, SQ2, robustesses ACE et cycle, décision d'étage.
- Mécanisme 7DK2cBQeHN, plateau, placebo pays par pays, confondant ACE, re-test contrôles propres.
- Plan de recherche pJHjkMZIYi et sous-pages de phases sous Results.

## Données et scripts internes
- Extractions, data/raw/firm_panel_2010.parquet, listed_panel_2010.parquet, ownership_flags.parquet. Publiées, data/dividends_cross_country.csv, dividends_cross_country_broad.csv, dividends_2010_2022_broad.csv, dividends_world.csv, household_flows_fr.csv, house_prices.csv, unemployment_2010_2022.csv, dividend_tax_rate_2010_2022.csv.
- Acquisition, data_download/orbis_panel.py, orbis_fr_coverage.py, ownership_flags.py, compustat_listed.py, dividends_cross_country.py, dividends_broad.py, household_destination.py, dividends_world.py.
- Venvs, .venv-repro pour la reproduction, .venv-acq pour l'acquisition WRDS et le CS.

## Ouvert
- Intégration finale du manuscrit, commencer par l'architecture pour ne pas noyer la contribution sous les robustesses, puis rédaction fine et tableaux au format paper-tables-format.
- Table des statistiques descriptives en annexe à générer par le code.
- Vérification bibliographique, email de contact.
