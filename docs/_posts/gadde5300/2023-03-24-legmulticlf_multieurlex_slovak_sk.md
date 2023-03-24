---
layout: model
title: Legal Multilabel Classification (MultiEURLEX, Slovak)
author: John Snow Labs
name: legmulticlf_multieurlex_slovak
date: 2023-03-24
tags: [legal, classification, sk, licensed, multieurlex, open_source, tensorflow]
task: Text Classification
language: sk
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: MultiClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Multilabel Text Classification model that can help you classify 16 types of Slovak legal documents.

## Predicted Entities

`pristúpenie k Európskej únii`, `maloobchod`, `vyšetrovací výbor`, `obchod medzi Východom a Západom`, `sprenevera`, `pridelenie zákazky`, `medzinárodný obchod`, `vnútorný obchod`, `zahraničný obchod`, `Aarhus (okres)`, `obchod so zbraňami`, `obchodovanie štátu`, `veľkoobchod`, `lepidlo`, `Komisia OSN`, `parlamentný výbor`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/legmulticlf_multieurlex_slovak_sk_1.0.0_3.0_1679671179887.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/legmulticlf_multieurlex_slovak_sk_1.0.0_3.0_1679671179887.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")\
    .setCleanupMode("shrink")

embeddings = nlp.UniversalSentenceEncoder.pretrained()\
    .setInputCols("document")\
    .setOutputCol("sentence_embeddings")

docClassifier = nlp.MultiClassifierDLModel().pretrained('legmulticlf_multieurlex_slovak', 'sk', 'legal/models')\
    .setInputCols("sentence_embeddings") \
    .setOutputCol("class")

pipeline = nlp.Pipeline(
    stages=[
        document_assembler,
        embeddings,
        docClassifier
    ]
)

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)

light_model = nlp.LightPipeline(model)

result = light_model.annotate("""Nariadenie Rady (ES) č. 2229/2003
z 22. decembra 2003,
ktorým sa uplatňuje konečné antidumpingové clo a s konečnou platnosťou sa vyberá dočasné clo uplatnené na dovozy kremíka z Ruska
RADA EURÓPSKEJ ÚNIE
so zreteľom na Zmluvu o založení Európskeho spoločenstva,
so zreteľom na nariadenie Rady (ES) č. 384/96 z 22. decembra 1995 o ochrane pred dumpingovými dovozmi z nečlenských krajín Európskeho spoločenstva [1], a najmä na jeho články 9 a 10 ods. 2,
so zreteľom na návrh predložený Komisiou po porade s Poradným výborom,
keďže:
1. Postup
1.1. Dočasné opatrenia
(1) Komisia nariadením (ES) č. 1235/2003 [2] (ďalej len "dočasné nariadenie") uplatnila dočasné antidumpingové clá na dovozy kremíka pochádzajúceho z Ruska, vo výške cla ad valorem v rozpätí medzi 24,0 % a 25,2 %.
(2) Pripomíname, že obdobie vyšetrovania dumpingu a ujmy trvalo od 1. októbra 2001 do 30. septembra 2002 (ďalej len "obdobie vyšetrovania" alebo "OV"). Skúmanie analýzy trendov v kontexte vyšetrovania vážnej ujmy trvalo od 1. januára 1998 do konca OV (ďalej len "obdobie skúmania").
1.2. Súčasné opatrenia
(3) V súčasnosti sa uplatňujú antidumpingové clá, vo výške 49 % sadzby ad valorem, na dovozy kremíka pochádzajúceho z Čínskej ľudovej republiky (ďalej len "Čína") [3], prehľad opatrení [4] podľa článku 11 ods. 2 základného nariadenia bude vydaný.
1.3. Následný postup
(4) Následne po uplatnení dočasného antidumpingového cla dostali záujmové strany základné fakty a stanoviská, na základe ktorých sa vydalo dočasné nariadenie. Niektoré strany predložili svoje písomné stanoviská. Komisia poskytla možnosť záujmovým stranám, ktoré o to požiadali, aby boli vypočuté
(5) Všetky strany boli informované o základných faktoch a zisteniach, na základe ktorých sa navrhovalo odporučiť uplatnenie konečného antidumpingového cla a výber s konečnou platnosťou čiastok zabezpečených ako dočasné clo. Taktiež im bol stanovený termín, do ktorého mohli po tomto zistení predložiť svoje stanoviská.
(6) Ústne a písomné argumentácie predložené týmito osobami boli zvážené a, ak to bolo vhodné, predbežné stanoviská sa náležite upravili.
(7) Komisia pokračovala v hľadaní a overovaní všetkých informácii, ktoré považovala za potrebné pre konečné stanoviská.
(8) V nadväznosti na overovacie návštevy uverejnené v odseku 7 dočasného nariadenia a po uplatnení dočasných opatrení, ďalšie overovacie návštevy boli vykonané v priestoroch nasledovných užívateľov v spoločenstve:
- GE Bayer Silicones, Leverkusen, Nemecko,
- Raffinera Metalli Capra SpA, Brescia, Taliansko,
- Vedani Carlo Metalli SpA, Miláno, Taliansko.
2. Predmetný výrobok a podobný výrobok
2.1. Predmetný výrobok
2.1.1. Stanoviská exportujúcich výrobcov
(9) V úvodnom ustanovení 9 dočasného nariadenia sa opisuje predmetný výrobok ako kremík, KN kód 28046900. Niektorí exportéri namietali proti zaradeniu kremíkových výparov, ktoré sú vedľajším produktom získaným filtráciou pri výrobe kremíka, do súčasného konania.
(10) Malo by sa poznačiť, že kremíkové výpary nezodpovedajú definícií predmetného výrobku podľa úvodných ustanovení 9 a 10 dočasného nariadenia, ak je zrejmé, že boli získané ako vedľajší produkt výroby kremíka vo forme prášku, ktorý sa používa ako aditívum. Preto sa potvrdzuje, že tento produkt, KN kód 28112200, je vyňatý z konania.
(11) Jeden ruský exportujúci výrobca namietal proti definícii výrobku, nakoľko v skutočnosti existujú v rámci tohto KN kódu dva rôzne typy kremíka, jeden určený na použitie v metalurgii a druhý na chemické účely. Na podporu svojho tvrdenia ruský výrobca vysvetľuje, že oba výrobky majú rôzne chemické zloženie, vzhľadom na obsah stopových prvkov, a sú pre rôznych koncových užívateľov; existujú dve rôzne skupiny užívateľov, ktoré nie sú porovnateľné; a že neexistuje možnosť výmeny týchto dvoch typov kremíka.
(12) Vyšetrovanie ukázalo, že kremík vyrábaný podľa rôzneho typu a kremík predávaný na trhu spoločenstva počas obdobia vyšetrovania, vyrábaný priemyselným odvetvím spoločenstva, alebo dovážaný z Ruska, obsahuje viac ako 95 % váhy. Typ kremíka je v prvom rade determinovaný ako percento kremíka a v druhom rade inými prvkami, spravidla obsahom irídia a vápnika. Pre špeciálnych užívateľov, najmä z chemického priemyslu, pomer stopových prvkov určuje využitie kremíka. Všeobecne je kremík pre špeciálnych užívateľov vyrábaný na základe špeciálnych požiadaviek a nakupuje sa po dlhom overovacom procese od individuálnych užívateľov.
(13) Dokázalo sa aj to, že vysoko tepelne odolný materiál nebol predávaný výlučne na chemické účely, a že aj užívatelia z chemického priemyslu nakupovali určité množstvá menej tepelne odolného, tzv. metalurgického kremíka. Je všeobecne dokázané, že užívatelia s požiadavkami na nižšiu kvalitu, ako aj metalurgickí užívatelia sú schopní používať vysoko tepelne odolný kremík. Pre nich je hlavný determinujúci faktor cena, preto nechcú platiť dodatočnú prémiu za viac tepelne odolný kremík, ako je ten, ktorý požadujú.
2.1.2. Stanoviská užívateľov
(14) Niekoľko užívateľov namietalo proti dočasnému určeniu daného výrobku. Stanoviská sa veľmi podobajú tým zaslaným od exportujúcich výrobcov, najmä metalurgických užívateľov. Všetci metalurgický užívatelia uvádzajú tri rôzne typy výrobku, o. i. chemického, a rozdiely v štandardoch a v nízko tepelne odolnom kremíku určenom pre metalurgických užívateľov. Všetci však akceptovali využitie oboch typov kremíka pri ich výrobe, aj keď uprednostňovali nízko tepelne odolný kremík kvôli nižším nákladom. Tieto stanoviská boli poslané aj organizáciami metalurgických užívateľov.
(15) Jeden chemický užívateľ sa vyjadril k zaradeniu výrobku. Potvrdil, že kremík, ktorý nakupuje, je vyrobený na mieru podľa špecifikácie a stopové prvky v kremíka sú najdôležitejším faktorom.
2.1.3. Stanoviská priemyselného odvetvia spoločenstva
(16) Priemyselné odvetvie spoločenstva súhlasí s dočasným zaradením všetkých typov kremíka podľa definície úvodných ustanovení 9 a 10 dočasného nariadenia. Poukázali na to, že veľa argumentov bolo použitých pre určenie podobného výrobku a nie predmetného výrobku a exportujúci výrobcovia si ich mýlia.
2.1.4. Záver ohľadne predmetného výrobku
(17) Kremík je výrobok vyrábaný v rôznych typoch, závislých v prvom rade od typu železa, v druhom rade od obsahu vápnika a v treťom rade od ostatných stopových prvkov. Výrobný proces v EÚ a v Rusku, o. i. vo vysokej peci pod elektrickým oblúkom, je preto do značnej miery rovnaký.
(18) Na trhu EÚ sú dve základné skupiny užívateľov: chemickí užívatelia vyrábajúci kremík a metalurgickí užívatelia vyrábajúci hliník. Metalurgickí užívatelia môžu byť ešte rozdelení do dvoch podskupín medzi primárnych výrobcov hliníka a sekundárnych výrobcov využívajúcich recyklovaný hliník. Používaný kremík, váhy minimálne 95 %, je však typický 98 % alebo 99 % kremík.
(19) Boli identifikované tri typy kremíka: vysoko tepelne odolný, stredne odolný (tzv. štandardný) a nízko tepelne odolný kremík v závislosti od percenta železa a vápnika v kremíku. Medzi týmito typmi bolo zistené prekrývanie vo využití rôznymi skupinami užívateľov. Bolo všeobecne akceptované, že neexistuje žiadna fyzická, chemická ani technická vlastnosť, ktorá by prekážala sekundárnym výrobcom hliníka pri využívaní všetkých typov kremíka, alebo primárnych výrobcov hliníka pri využívaní stredne odolného alebo vysoko tepelne odolného kremíka. Neexistuje rovnaká úroveň vzájomnej výmeny medzi oboma stranami, aj keď bol poskytnutý dôkaz od chemických užívateľov, že používajú stredne odolný alebo nízko tepelne odolný kremík. Náklady na rôzne typy určujú, ktorý typ bude využívať ktorá skupina užívateľov.
(20) Vyšetrovanie ukázalo, ako bolo hore uvedené, že všetky typy kremíka, napriek rozdielom v požiadavkách na obsah a na iné chemické prvky, majú rovnakú základnú chemickú, fyzikálnu a technickú charakteristiku. Aj keď kremík sa môže používať na rôzne konečné účely, bolo zistené, že existuje substitučnosť medzi nízko odolným a vysoko tepelne odolným typom a rozdielnym použitím.
(21) Zistenia podľa úvodných ustanovení 9 a 10 dočasného nariadenia sú týmto definitívne potvrdené.
2.2. Podobný výrobok
(22) Po analýze bolo zistené, že námietka proti kontrolnému výrobnému číslu (PCN) uvedenému v odseku 14 dočasného nariadenia sa vzťahuje na porovnanie ceny kremíka vyrobeného v Rusku a v spoločenstve a tomu zodpovedajúcu úroveň eliminácie ujmy. Rozdiely v cenách, kvalite a v použití nevedú nutne k záveru, že nejde o podobný produkt. V skutočnosti je zrejmé, že v tomto kontexte boli použité rôzne typy výrobku s rovnakými základnými fyzikálnymi a chemickými vlastnosťami. Musíme vziať do úvahy vyššie uvedené rozdiely medzi exportnou cenou a bežnou hodnotou a v determinácií napríklad cenového podliezania a úrovne eliminácie ujmy.
(23) Jeden exportujúci ruský výrobca informoval o antidumpingových opatreniach uplatňovaných v súčasnosti na dovoz kremíka z Číny (pozri úvodné ustanovenie 3). V skutočnosti odkazuje na úvodné ustanovenie 55 nariadenia Rady č. 2496/97, v ktorom sa uvádza: "kvalita kremíkového kovu z Ruska a Ukrajiny nie je porovnateľná s európskym alebo čínskym kremíkovým kovom".
(24) V odpovedi na tento bod je potrebné v prvom rade uviesť, že tento opis sa vzťahuje na vyšetrovanie uskutočnené pred viac ako piatimi rokmi a to nezodpovedá súčasnej situácii. V skutočnosti sa odsek 55 hore uvedeného nariadenia zaoberá iba príčinnou súvislosťou. Z formulácie je jasné, že predmetný a podobný výrobok zo všetkých zdrojov, z Číny, Ruska, EÚ, alebo analogickej krajiny, napr. Nórska, je kremík. Tieto typy kremíka zodpovedajú definícii podobného výrobku podľa článku 1 odsek 4 základného nariadenia. Navyše do veľkej miery existujú rozdiely v kvalite medzi rôznymi výrobcami kremíka v rôznych krajinách a tieto rozdiely musia byť vzaté do úvahy pri rozhodovaní. Poznamenávame, že boli zistené rozdiely v kvalite v rámci vývozu rôznych typov výrobku z Ruska do spoločenstva.
(25) Na základe zistení z vyšetrovania a hore uvedených faktov sa potvrdzuje, že kremík vyrábaný v Rusku pre domáci trh a určený na export do spoločenstva, kremík predávaný na domácom trhu v analogickej krajine a produkt vyrábaný odvetvím spoločenstva a predávaný v spoločenstve, majú rovnaké základné fyzikálne a chemické vlastnosti. Z toho sa vyvodzuje záver, že všetky typy kremíka sú jedným výrobkovým radom a zodpovedajú tým definícií podobného výrobku podľa článku 1 odseku 4 základného nariadenia.
3. Dumping
3.1. Cena v bežnom obchode
(26) Na základe chýbajúcich stanovísk k úvodným ustanoveniam 15 a 18 o prístupe k trhovému hospodárstvu, sa tieto úvodné ustanovenia dočasného nariadenia potvrdzujú.
(27) Všetci exportujúci výrobcovia poskytli stanoviská, v ktorých argumentujú doplnením nákladov na elektrickú energiu do dočasného konania. Zdôrazňujú, že ich hlavný dodávateľ elektrickej energie je spoločnosť v súkromnom vlastníctve a jej nízke ceny je možné vysvetliť komplexom prepojených hydro-energetických elektrární rozmiestnených po celom svete, ktoré využívajú komparatívne výhody. Tieto okolnosti boli už skôr predmetom vyšetrovania, ale keď sa zistilo, že ceny elektrickej energie v Rusku sú regulované a tým je cena poskytovanej elektrickej energie nízka oproti podobným poskytovateľom elektrickej energie vyrobenej z hydro-energetických elektrární v analogických krajinách Nórsku a Kanade, rozhodlo sa o zamietnutí tejto požiadavky a potvrdilo sa predbežné rozhodnutie použiť cenu elektrickej energie stanovenú iným dodávateľom z Ruska. Táto cena bola zistená v zmysle najnižších cien producentov elektrickej energie v spoločenstve.
(28) Na základe chýbajúcich stanovísk k odsekom 19 a 26 o určení ceny v bežnom obchode, sa tieto odseky dočasného nariadenia potvrdzujú.
3.2. Vývozná cena
(29) Všetci exportujúci výrobcovia argumentovali, že spoločnosti so sídlom mimo Ruska, zahrnuté medzi predávajúcich na území spoločenstva, sú záujmové strany a teda by mali byť pripojené ako individuálna ekonomická jednotka spolu so spoločnosťami so sídlom v Rusku. Zdôrazňujú, že použitá exportná cena by mala byť cena výrobku dodaného prvému nezávislému spotrebiteľovi v ES.
(30) V prípade výrobcu so sídlom v spoločenstve (Veľká Británia) nebol poskytnutý žiadny nový dôkaz, ktorý by poukazoval na jeho spojenie s exportujúcim výrobcom. Námietka sa preto zamietla a dočasný nárok na konštruovanie exportnej ceny na základe predajných cien tomuto dovozcovi bol potvrdený.
(31) V prípade výrobcu so sídlom vo Švajčiarsku na základe overovacej návštevy vykonanej pred uplatnením dočasného opatrenia sa zistilo, že spoločnosť spolupracovala s exportujúcim výrobcom. Predaj uskutočňovaný prostredníctvom tohto dovozcu sa realizoval na základe exportnej ceny pre prvého nezávislého spotrebiteľa v spoločenstve.
(32) K prípadu dovozcu so sídlom na britských Panenských ostrovoch v prvom rade poznamenávame, že podľa článku 2 ods. 8 základného nariadenia je exportná cena definovaná ako: "cena aktuálne platená alebo splatná pri predaji výrobku určeného na export do spoločenstva". Inými slovami, v prípade, že do exportnej operácie do spoločenstva sú zainteresovaní sprostredkovatelia, nie je to cena zaplatená spotrebiteľom v spoločenstve (ktorý nie je v kontakte s exportujúcim výrobcom), ale cena, pri ktorej výrobok "opúšťa" exportujúcu krajinu. Táto cena môže byť nahradená inou v prípade, že sa strany navzájom poznajú. Rusal poskytol novú informáciu, podľa ktorej z ich pohľadu potvrdzujú partnerstvo. Je však dokázané, že toto partnerstvo nebolo jednoznačne preukázané. V skutočnosti neexistuje priamy akciový podiel spoločnosti Rusal v spoločnosti na britských Panenských ostrovoch a takisto sú ich štruktúry zložité a netransparentné. Podľa vyjadrenia spoločnosti je prepojenie dosiahnuté nepriamym akciovým podielom, ale v tejto súvislosti neboli poskytnuté žiadne relevantné dokumenty. Navyše, podľa spoločnosti Rusal spoločnosť na britských Panenských ostrovoch nevykazuje žiadnu ekonomickú aktivitu v predaji a exporte výrobkov, ale funguje ako korešpondenčná spoločnosť. Inými slovami sa nejedná o predaj prostredníctvom tretej strany. Spoločnosť na britských Panenských ostrovoch je skôr adresát s nejasnými účtovnými cieľmi. Neexistoval spôsob overiť skutočnú úlohu tejto spoločnosti so sídlom na britských Panenských ostrovoch alebo preskúmať jej finančné toky. Nakoniec sa rozhodlo o potvrdení dočasného prístupu a na konštruovaní exportnej ceny na základe predajnej ceny spoločnosti na britských Panenských ostrovoch.
3.3. Porovnanie
(33) Jeden exportujúci výrobca znova žiadal úpravu fyzikálnych vlastností založenú na fakte, že priemerný stupeň tepelnej odolnosti kremíka predávaného na území Ruska je vyššej kvality, v dôsledku čoho vznikajú vyššie výrobné náklady. Spoločnosť však nebola schopná dokázať rovnaké zloženie výrobku predávaného v Rusku a na území spoločenstva, z tohto dôvodu námietka proti fyzikálnym vlastnostiam bola zamietnutá a potvrdil sa dočasný prístup.
(34) Dve spoločnosti sa sťažovali na množstvo a úroveň obchodu. Požiadavka na množstvo nemôže byť akceptovaná, pretože spoločnosť nepredložila diskonty a zľavy pre určité množstvá pri určitom objeme obchodu a pre určitých zákazníkov uznané pri dočasnom postupe. Vzhľadom na požiadavku o prehodnotenie úrovne obchodu spoločnosť nebola schopná dokázať, že odhad vykonaný v dočasnom postupe je nedostatočný a preto nie je možné dodatočné prehodnotenie.
3.4. Dumpingové rozpätie pre vyšetrované spoločnosti
(35) Pre absenciu pripomienok sa potvrdilo dumpingové rozpätie podľa úvodných ustanovení 29 a 30 dočasného nariadenia.
(36) konečné dumpingové rozpätia v percentách dovoznej ceny CIF na hranici spoločenstva sú takéto:
Spoločnosť | Dumpingové rozpätie |
OJSC "Bratsk Aluminium Plant" (RUSAL Group) | 23,6 % |
SKU LLC, Sual Kremny - Ural i ZAO Kremny (SUAL Group) | 24,8 % |
Rusko | 24,8 % |
4. Vážna ujma
4.1. Priemyselné odvetvie spoločenstva
(37) Pre absenciu pripomienok k definícii priemyselného odvetvia spoločenstva sa týmto potvrdzuje obsah a závery úvodných ustanovení 33 a 34 dočasného nariadenia.
4.2. Spotreba kremíka v spoločenstve
(38) Pre absenciu nových informácii o spotrebe sa týmto potvrdzujú dočasné zistenia podľa úvodných ustanovení 35 a 36 dočasného nariadenia
4.3. Dovoz kremíka do spoločenstva
4.3.1. Objem a trhový podiel na dovoze
(39) Pre absenciu nových informácii o dovoze kremíka do spoločenstva alebo o trhovom podiele, sa týmto potvrdzujú dočasné zistenia podľa úvodných ustanovení 37 a 43 dočasného nariadenia.
4.3.2. Cenové podliezanie a stlačenie
(40) Vypočítanie cenového podliezania bolo revidované vzhľadom na kvalitu a úroveň obchodu.
Táto úprava bola vykonaná na základe overenej informácie a korešponduje s predbežným výpočtom trhovej hodnoty cenových rozdielov.
(41) Konečné cenové podliezanie podľa cenového rozpätia predstavuje 10,2 %.
(42) Na existenciu a úroveň cenového podliezania by mala malo nazerať s prihliadaním na skutočnosť, že ceny boli stlačené. Počas sledovaného obdobia boli ceny stlačené o 16 %, čo nepokrýva rozsah úplných nákladov na výrobu v odvetví spoločenstva počas obdobia vyšetrovania.
4.4. Hospodárska situácia v odvetví spoločenstva
(43) Dvaja ruskí exportujúci výrobcovia namietali proti tomu, že odvetvie spoločenstva neutrpelo materiálnu ujmu, lebo väčšina indikátorov ujmy vykazuje pozitívny vývoj. V skutočnosti exportujúci výrobcovia
(44) Indikátory ujmy odvetvia spoločenstva, ako je uvedené v úvodných ustanoveniach 71 a 72 dočasného nariadenia, preukazujú pozitívny vývoj najmä v rokoch 1998 a 2000. Medzi rokom 2000 a obdobím vyšetrovania všetky indikátory stúpali pomaly alebo stagnovali, prípadne niektoré klesali. Počas tohto obdobia je očividná materiálová ujma odvetvia spoločenstva.
(45) Je potrebné poznamenať, ako bolo naznačené v úvodnom ustanovení 72 dočasného nariadenia, že relatívne dobré výsledky odvetvia spoločenstva pred rokom 2000 boli spôsobené rozhodnutím odvetvia investovať do dodatočných výrobných zdrojov. Počas tohto obdobia vzrástla
(46) Následne v zhode s nárastom dumpingového dovozu z Ruska sa zhoršila situácia v odvetví spoločenstva. Trhový podiel, cash-flow, investície a návratnosť investícií vykázali hlboký pokles.
(47) Napokon, trendy ostatných indikátorov ujmy, súčasne s poklesom ziskovosti a znížením predajných cien v odvetví spoločenstva, vedú k záveru, že odvetvie spoločenstva utrpelo značnú ujmu.
4.5. Záver o ujme
(48) Vzhľadom na tieto výsledky a pre absenciu nových informácii, ktoré by vyžadovali prehodnotenie zistení, že odvetvie spoločenstva utrpelo počas obdobia vyšetrovania vážnu materiálnu ujmu, najmä v oblasti cien a ziskovosti; argumenty predložené exportujúcimi ruskými výrobcami sa zamietajú. Zistenia a závery podľa úvodných ustanovení 71 a 73 dočasného nariadenia sa týmto potvrdzujú.
5. Príčinná súvislosť
(49) Jeden ruský exportujúci výrobca namietal, že aj keď sa potvrdila značná ujma, táto nebola spôsobená dovozom ruského kremíka. Celý rad ďalších faktorov sa označuje ako príčina ujmy, ak k nejakej došlo, ktorú utrpelo odvetvie spoločenstva. Ostatné tretie krajiny s oveľa väčším podielom na dovoze v porovnaní s Ruskom, ujma odvetviu spoločenstva, ktorú si spôsobilo samo, exportné schopnosti odvetvia spoločenstva, dovoz kremíka odvetvím spoločenstva a rozdiely na trhu s metalurgickým a chemickým kremíkom, boli citácie ako príklad pre spôsobenie vážnej ujmy odvetvia spoločenstva. Jeden ruský výrobca tiež poukázal na 16 % cenový rozdiel medzi cenami odvetvia spoločenstva a ruskými cenami počas obdobia vyšetrovania a že vzhľadom na tento veľký cenový rozdiel neexistuje cenová konkurencia medzi kremíkom z rôznych zdrojov na trhu spoločenstva.
5.1. Dovoz z ostatných tretích krajín
(50) Ako je uvedené v úvodnom ustanovení 98 dočasného nariadenia, dovoz z niektorých tretích krajín preukázal podstatne vyšší nárast ako dovoz z Ruska. Dovoz z každej z týchto krajín, s výnimkou Číny, klesol na objem dovozov medzi rokom 2000 a obdobím vyšetrovania, aj keď odvetvie spoločenstva zaznamenalo zhoršenie svojej hospodárskej situácie. Okrem toho ceny týchto dovozov boli vo všetkých prípadoch vyššie ako tie, ktoré sa realizovali z Ruska a aj keď cenovo podliezli ceny odvetvia spoločenstva, cenový rozdiel bol len minimálny.
(51) Jeden ruský exportujúci výrobca namietal, že informácie z Eurostatu nemôžu byť spoľahlivé, ak neberú do úvahy rozdiely v produktovom mixe. Zdôrazňoval, že sú dôležité cenové rozdiely medzi kremíkom s prevažne nižšou kvalitou exportovaným z Ruska a kremíkom s vyššou kvalitou z ostatných tretích krajín. Namietal, že keď sa porovnávajú ceny, mala by sa použiť skôr cena aktuálne platená užívateľmi kremíka pochádzajúceho z rôznych zdrojov.
(52) Tento výrobca však neuviedol žiadne dôkazy na podporu svojej námietky. Navyše, kvôli nedostatočným základným údajom o cenách platených užívateľmi z ostatných tretích krajín sa nemohlo vykonať porovnanie cien. Informácie získané z Eurostatu za týchto okolností reprezentujú najlepší možný zdroj pre určenie cien z ostatných tretích krajín. Vzhľadom na informácie získané z paralelne ukončeného preskúmania vedeného proti Číne, bolo zistené priemerné rozpätie cenového podlezenia, ak sa porovnali ceny na základe rôznych stupňov, čo bolo v súlade s priemerným rozpätím, ak sa porovnali priemerné ceny odvetvia spoločenstva s priemernými dovoznými cenami podľa Eurostatu.
(53) Tiež by malo byť poznamenané, že pre správne porovnanie dovozných cien boli údaje z Eurostatu použité pri všetkých prípadoch. Napríklad pre Rusko bola dostupná počas obdobia vyšetrovania overená informácia a správna cena bola o niečo nižšia ako tá získaná z Eurostatu.
5.2. Ujma spôsobená vlastným zavinením
(54) Namietalo sa, že ujma spôsobená odvetviu spoločenstva bola primárne zapríčinená nárastom nákladov vzniknutých pri budovaní nových výrobných kapacít s cieľom dosiahnuť trhový podiel. Na podporu toho sa namietalo, že priemysel spoločenstva má najvyššie priemerné výrobné náklady (ďalej len "VN") na svete. Námietka bola podporená údajmi o overených VN priemyslu spoločenstva a ruského priemyslu, zistených pri vyšetrovaní, oproti uverejneným údajom o nákladoch z ostatných tretích krajín. Základné náklady zahrnuté do uverejnených štatistických údajov neboli však jasne identifikované a preto neexistuje dôkaz, že by tieto údaje mohli byť porovnávané s overenými údajmi o VN zistenými počas vyšetrovania. Pre tieto údaje je typické, že obsahujú len niektoré výrobné náklady a niektoré základné prvky nákladov, ako napríklad predaj, všeobecné a administratívne náklady. Treba ešte dodať, že ruský výrobca neposkytol výrobcom spoločenstva žiadne tomu zodpovedajúce uverejnené údaje. Na základe toho sa zistilo, že námietka nemôže byť adresovaná a argumenty udávané ruským výrobcom sa zamietli. Na podporu tohto prístupu sa použili overené VN analogickej krajiny, Nórska, a VN boli vyššie ako tie, ktoré uviedol ruský výrobca. Ak sa použijú celkové náklady, overené VN v Nórsku sú porovnateľné s VN priemyslu spoločenstva.
(55) Aj keď by boli náklady spoločenstva komparatívne vyššie, tento fakt by nemohol vyvrátiť spojenie medzi nízko cenovým dumpingovým dovozom a ujmou spôsobenou priemyselnému odvetviu spoločenstva. Ako bolo spomenuté v úvodnom ustanovení 83 dočasného nariadenia, ak by neklesli ceny medzi rokom 2000 a obdobím vyšetrovania, priemysel spoločenstva by zaznamenal zisk na úrovni 1,7 % a nie aktuálnu stratu vo výške 2,1 %.
5.3 Vývozy priemyselného odvetvia spoločenstva
(56) Namietalo sa, že zníženie predaja odvetvia priemyslu spoločenstva pri vývoze by malo dopad na ich ziskovosť pri predaji v EÚ. Neexistuje však dôkaz na podporu tejto námietky.
(57) Celkový pokles predaja medzi rokom 1998 a obdobím vyšetrovania bol iba 2,3 % z celkového predaja odvetvia spoločenstva. Dopad na ceny a ziskovosť odvetvia spoločenstva (ak nejaký vôbec bol) bol veľmi zanedbateľný. Takisto sa môže tvrdiť, že pokles exportu bol čiastočne spôsobený dopytom po kremíka vyrobenom počas obdobia vyšetrovania odvetvím spoločenstva.
5.4 Dovoz kremíka odvetvím spoločenstva
(58) Jeden ruský výrobca vzniesol otázku ohľadne záveru úvodného ustanovenia 85 dočasného nariadenia, že spoločnosti kupujúce kremík, ktoré sú spriaznené s priemyselným odvetvím spoločenstva, prijali také rozhodnutie vo svojom mene a bez vplyvu zo strany priemyslu spoločenstva. Na podporu toho tvrdil, že tieto spriaznené spoločnosti nemali umožnené vyjadriť svoj názor na konanie. Toto tvrdenie má dokázať, že tieto spoločnosti sú naozaj ovládané odvetvím spoločenstva.
(59) Skutočnosť, že spoločnosti spriaznené s priemyslom spoločenstva neposkytli stanoviská, ktoré by oponovali antidumpingové opatrenia v tomto konaní neznamená, že nemôžu voľne čerpať suroviny na základe finančných vzťahov. Keďže tieto spoločnosti dokázateľne nakupovali kremík z priemyslu spoločenstva, z Ruska ako aj z ostatných tretích krajín podľa toho, ako sa rozhodli, záver úvodného ustanovenia 85 dočasného nariadenia je preto potvrdený.
5.5. Rozdiely na trhoch s chemickým a metalurgickým kremíkom
(60) Tvrdilo sa, že problémy, ktorým priemysel spoločenstva čelí od roku 2000 až podnes, sú spôsobené znížením dopytu po chemickom type kremíka zapríčinenom poklesom dopytu po výrobkoch chemického priemyslu. Namietalo sa, že odvetvie spoločenstva predáva kremík vo vyššej miere chemickým užívateľom ako metalurgickým, čo je opakom ako u ruských exportujúcich výrobcov. Keďže ruský kremík na chemickom trhu nekonkuruje kremíku vyrobenému odvetvím spoločenstva, problémy odvetvia spoločenstva preto nemožno pripísať na konto ruského dovozu.
(61) Nižšie uvedená tabuľka vyjadruje cenové trendy a objem predaja kremíka odvetvím spoločenstva chemickým zákazníkom.
Predaj kremíka odvetvím spoločenstva chemickým zákazníkom
Zdroj:
priemyselné odvetvie spoločenstva
| 1998 | 1999 | 2000 | 2001 | OV |
Tony | 48907 | 59924 | 74880 | 74435 | 69652 |
Index | 100 | 123 | 153 | 152 | 142 |
EUR/tona | 1488 | 1313 | 1287 | 1316 | 1301 |
Index | 100 | 88 | 86 | 88 | 87 |
(62) Z tabuľky vyplýva, že počas hodnoteného obdobia stúpol predaj kremíka chemickým užívateľom v objeme o 42 %, pričom priemerná cena poklesla o 13 %. Toto je porovnávané s 57 % nárastom objemu a 16 % poklesom cien pri všetkých predajoch kremíka počas hodnoteného obdobia (pozri tabuľku 8 a 9 v dočasnom nariadení).
(63) Počas obdobia medzi rokom 2000 a OV, keď trendy ujmy poukázali na pokles cien a ziskovosti, predaj chemickým užívateľom klesol asi o päťtisíc ton (- 7 %), ale priemerné ceny stúpli o 14 EUR/tona (+ 1,1 %). Porovnateľné údaje za celkový predaj ukazujú nárast o približne tritisíc ton (+ 2,1 %), pričom priemerné ceny klesli o 46 EUR/tona (-3,7 %).
(64) Neexistuje preto dôvod veriť, že ujma spôsobená priemyselnému odvetviu spoločenstva bola zapríčinená poklesom predaja chemickým užívateľom. V skutočnosti, vzhľadom na charakter utrpenej ujmy, je opak pravdou.
(65) Následkom toho sa zamieta námietka, že práve trend predaja chemickým užívateľom, ktorý odvetvie spoločenstva zaznamenáva, bol skutočným dôvodom utrpenej ujmy.
5.6. Porovnanie cien
(66) Vzhľadom na rozdiel v cene medzi kremíkom vyrobeným v spoločenstve a kremíkom vyrobeným v Rusku je potvrdené, že rozdiel nie je 16 %, ako tvrdil ruský exportér, ale v priemere 11 % počas OV (pozri úvodné ustanovenie 46 dočasného nariadenia). Tento rozdiel existoval napriek poklesom cien v odvetví spoločenstva vo výške 7 % v období medzi rokom 2001 a OV. To jasne poukazuje na vplyv ruských cien na ceny v odvetví spoločenstva. Tvrdenie, že cenové podliezanie je také vysoké, že nemohlo spôsobiť ujmu priemyselnému odvetviu spoločenstva, možno považovať za intuitívne.
(67) Vyšetrovanie v skutočnosti ukázalo, že veľké množstvo kremíka od výrobcov v spoločenstve ako aj od ruských výrobcov sa predávalo rovnakým zákazníkom, prípadne rovnakým sprostredkovateľom tohto sektora. Nízka úroveň cien ruských exportérov jasne ukazuje na využívanie cenovej hladiny pri rokovaniach s priemyselným odvetvím spoločenstva.
5.7. Záver vzhĽadom na príČinnú súvislosŤ
(68) Vzhľadom na hore uvedené, námietky vyjadrené ruským exportujúcim výrobcom sa zamietajú a potvrdzujú sa závery a zistenia podľa odseku 101 a 102 dočasného nariadenia.
6. Záujem spoloČenstva
(69) Vzhľadom na dočasné zistenie, že uplatnenie opatrení nie je v rozpore so záujmami spoločenstva, boli záujmové strany vyzvané na spoluprácu pri konaní. Štyria užívatelia a užívateľská asociácia, ktorá spolupracovala pri vyšetrovaní, poslali svoje stanoviská. Piati užívatelia a užívateľská asociácia, ktorá nespolupracovala pri vyšetrovaní, poslali svoje stanoviská k dočasným zisteniam. Svoje stanoviská neposlali žiadni dovozcovia kremíka. Traja dodávatelia surovín ruským výrobcom poslali svoje stanoviská k dočasnému obdobiu.
(70) Tieto stanoviská uverejnené v dočasnom nariadení mali len poukázať na rozdiel medzi chemickým a metalurgickým kremíkom v základných údajoch charakterizujúcich predmetný a podobný výrobok. Žiadny užívateľ neposlal stanovisko k vplyvu opatrení na svoje náklady alebo ziskovosť, ako ani potrebné informácie umožňujúce vypracovať takýto odhad.
(71) Na základe overovacích návštev vykonaných u užívateľov bolo zistené, že aj keď sú proti opatreniam, ktoré zvýšia ich náklady, všeobecne súhlasia s metódami použitými v analýze. Je pravdepodobné, že opatrenie bude mať vplyv na užívateľov. Získaná informácia naznačuje, že clo zvýši náklady metalurgických užívateľov pri finálnom výrobku o cca 11 EUR za tonu, čo predstavuje asi 0,8 %.
(72) Pre dodávateľov surovín do spoločenstva môže uplatnenie opatrenia pôsobiť negatívne, t. j. znížiť ich obrat a ziskovosť, aj keď neexistuje dôkaz, ktorý by vyvrátil očakávané zisky v odvetví spoločenstva.
(73) Vzhľadom na to, že neboli poskytnuté žiadne nové informácie, ktoré by viedli k záveru, že uplatnenie konečného opatrenia by bolo proti záujmu spoločenstva, záver podľa odseku 118 dočasného nariadenia sa definitívne potvrdzuje.
7. KONEČNÉ OPATRENIA
(74) Vzhľadom na závery o dumpingu, vážnej ujme, príčinnej súvislosti a záujme spoločenstva, potvrdzuje sa konečné antidumpingové opatrenie uplatňované ako prevencia pred dovozom z Ruska hroziacim spôsobiť škodu priemyselnému odvetviu spoločenstva.
7.1. ÚroveŇ odstránenia ujmy
(75) Pre určenie metodiky boli použité námietky voči výpočtu úrovne odstránenia ujmy v dočasnom období.
7.1.1. TabuĽka pcn
(76) Namietalo sa, že tabuľka PCN, uvedená v úvodnom ustanovení 14 dočasného nariadenia, ktorá identifikuje všetky typy kremíka, neobsahuje dostatočné podrobnosti o chemickom zložení rôznych typov kremíka, a preto nie je možné správne porovnať rôzne typy kremíka. Bolo navrhnuté, aby sa tabuľka PCN zmenila a doplnila tak, aby jasnejšie identifikovala všetky typy kremíka dovážaného z Ruska a kremíka predávaného priemyselným odvetvím spoločenstva.
(77) Jedna spoločnosť namietala, že ako extra stupeň by mal byť zahrnutý kremík s obsahom železa vyšším ako 0,8 %. Aj keď kremík s vyšším obsahom železa by mohol viesť k nižším cenám na trhu, nebol poskytnutý žiadny dôkaz, že existuje rozdiel na trhu s kremíkom s obsahom železa vyšším ako 0,5 % a kremíkom obsahujúcim viac ako 0,8 %. Výsledkom rôzneho obsahu železa v kremíku by mohol byť len rozdiel v cene, ako spôsob na prispôsobenie ceny. Preto sa táto námietka zamieta.
(78) Ostatní ruskí exportujúci výrobcovia žiadali dve zmeny v tabuľke PCN. Po prvé, žiadali definovanie nového typu vzhľadom na obsah stopových prvkov, ktoré sú určujúcim faktorom. Namietali, že bez tohto dodatku by kremík predávaný metalurgickým užívateľom bol nespravodlivo porovnávaný s kremíkom predávaným chemickým užívateľom. Žiadali aj to, aby kremík obsahujúci presne 0,5 % železa bol klasifikovaný ako nízko kvalitný, namiesto štandardného, ako je uvedené v tabuľke PCN.
(79) Akceptovanie prvej požiadavky by neviedlo k spresneniu tabuľky PCN, ale skôr k zhoršeniu definovaných kritérií, pričom by hrozilo, že záujmové strany by mali do istej miery voľnosť pri zaraďovaní kremíka do príslušnej tabuľky PCN. Takáto sloboda by znížila spoľahlivosť informácii uvedených v PCN a tým by znížila úroveň spoľahlivosti stanovenej úrovne odstránenia ujmy. Neexistuje ani dôkaz nasvedčujúci tomu, že súčasná štruktúra tabuľky PCN by viedla k mylným alebo menej presným zisteniam. Napríklad vyrovnávacia kalkulácia založená na nízkej a štandardnej kvalite kremíka by viedla k výsledku v maximálnou odchýlkou 0,2 %. Z týchto dôvodov sa námietka zamieta.
(80) Pokiaľ ide o druhú námietku, nebol poskytnutý žiadny dôkaz na jej podporu. Okrem toho existujú náznaky, že kremík obsahujúci 0,5 % železa užívatelia vnímajú ako štandardný stupeň. Preto nie je potrebné robiť v tabuľke PCN žiadne úpravy.
7.1.2. Ziskové rozpätie
(81) Dočasne sa zistilo, že vzhľadom na celkový obrat by ziskové rozpätie vo výške 6,5 % mohlo predstavovať príslušné minimum, ktoré očakáva priemyselné odvetvie spoločenstva na elimináciu škodlivého dumpingu. Namietalo sa, že toto rozpätie je príliš vysoké a že by bolo vhodnejšie rozpätie okolo 3 %.
(82) Požiadavka na 3 % rozpätie nie je založená na faktoch. Okrem toho je 6,5 % rozpätie v súlade so ziskom dosiahnutým odvetvím spoločenstva v prípade spravodlivých podmienok na trhu, ako napríklad v období rokov 1998 a 2000. Naviac je pravdepodobné, že ak sa určí úroveň dumpingového rozpätia podľa zistení, dosiahne odvetvie spoločenstva výšku zisku ako pred obdobím vyšetrovania.
7.1.3. Odhad kvality
(83) Jeden ruský výrobca namietal, že kremík vyrobený v jednom z jeho závodov má nižšiu kvalitu ako kremík vyrobený v inom závode, a to v dôsledku rôzneho procesu výroby. Navyše namietal, že kremík s nižšou kvalitou by sa mal upraviť, aby sa dosiahlo spravodlivé porovnanie s cenami priemyselného odvetvia spoločenstva. Námietka o úprave sa týkala rozdielu medzi priemernými výrobnými nákladmi dvoch výrobných závodov.
(84) Treba priznať, že medzi oboma závodmi existujú rozdiely v kvalite. Aby sa mohli vykonať úpravy, bolo treba demonštrovať, že tieto rozdiely majú rozdielny vplyv na ceny na trhu, v tomto prípade EÚ. Preto sa vykonalo porovnanie na základe rôznych typov, aby bolo vidieť konzistentný rozdiel v predajných cenách dosiahnutý medzi oboma závodmi. V prípade vysoko kvalitného kremíka sa nerealizovali žiadne predaje zo závodu vyrábajúceho nízko kvalitný kremík a neboli potrebné žiadne úpravy. V prípade štandardného kremíka bol zistený jasný cenový rozdiel a bola vykonaná 4 % úprava predaja z príslušného závodu. V prípade nízko kvalitného kremíka sa nezistili žiadne cenové rozdiely a tak neboli vykonané žiadne úpravy.
(85) Druhý ruský výrobca namietal, že kremík vyrobený v jeho závode je takej nízkej kvality, že ho nemožno priamo porovnávať s nízko kvalitným kremíkom vyrobeným priemyselným odvetvím spoločenstva.
(86) Znovu sa akceptuje, že úroveň obsahu železa je vo všeobecnosti vyššia v kremíku vyrobenom týmto výrobcom ako v kremíku vyrobenom priemyslom spoločenstva aj iným výrobcom v Rusku. V rámci kalkulácie efektu kvality pre tohto výrobcu sa pre cenu dosiahnuteľnú na trhu EÚ použilo porovnanie priemerných cien iného ruského výrobcu na báze porovnania jednotlivých typov.
(87) Výsledky porovnania preukázali, že by sa mali upraviť ceny nízko kvalitného kremíka ruského výrobcu, aby sa tieto ceny mohli porovnať s cenami nízko kvalitného kremíka vyrobeného priemyselným odvetvím spoločenstva.
7.1.4. ÚroveŇ úpravy obchodu
(88) Ruský výrobca požadoval úpravu ceny pre rôzne úrovne obchodu v rámci predaja v EÚ. Zistilo sa, že ruský výrobca predával všetok kremík prostredníctvom obchodníka na britských Panenských ostrovoch. Druhý ruský výrobca predával prostredníctvom spriazneného obchodníka vo Švajčiarsku, prostredníctvom nezávislého obchodníka v EÚ a aj priamo koncovým zákazníkom. Priemyselné odvetvie spoločenstva predávalo všetok svoj kremík priamo konečným používateľom.
(89) V rámci určenia, či bola úroveň úpravy obchodu správna, analyzovali sa všetky obchodné prepojenia na rovnakých stupňoch predaja u rôznych výrobcov, aby sa zistil správny cenový diferenciál. Výsledkom tejto analýzy je, že úroveň úpravy obchodu bola podporovaná predajom prostredníctvom nezávislého obchodníka.
7.2. Forma a úroveŇ cla
(90) Podľa článku 9 ods. 4 základného nariadenia by malo byť uplatnené konečné antidumpingové opatrenie vo výške zisteného dumpingového rozpätia alebo ujmy, podľa toho, čo je nižšie. Tieto opatrenia, ako aj dočasné opatrenia, by mali mať formu cla ad valorem.
7.3. Výber s koneČnou platnosŤou doČasného cla
(91) Vzhľadom na krivku dumpingového rozpätia zisteného u ruských exportujúcich výrobcov a na úroveň vážnej ujmy spôsobenej priemyselnému odvetviu spoločenstva sa považuje za nutné, aby sumy zabezpečené prostredníctvom dočasného antidumpingového cla uloženého dočasným nariadením č. 1235/2003 sa s konečnou platnosťou vybrali v rozsahu sumy uloženého konečného cla. Keď sú konečné clá vyššie ako dočasné, mali by sa s konečnou platnosťou vybrať iba sumy zabezpečené na úrovni dočasných ciel.
(92) Každý nárok na uplatnenie týchto osobitných firemných colných antidumpingových sadzieb (napr. po zmene mena subjektu alebo po vytvorení nových produkčných alebo výrobných subjektov) by mal byť adresovaný Komisii bezodkladne a so všetkými relevantnými údajmi, najmä o akejkoľvek zmene aktivít firmy súvisiacich s výrobou, domácimi tržbami a tržbami za vývoz súvisiacimi napr. so zmenou mena alebo zmenou výrobných alebo obchodných subjektov. Ak to bude primerané, nariadenie bude náležite upravené aktualizáciou zoznamu firiem požívajúcich osobitné clá.
7.4. Vyrovnávacie opatrenia
(93) V súvislosti s uplatnením dočasného opatrenia a po zverejnení konečných zistení jeden ruský exportujúci výrobca ponúkol vyrovnávaciu cenu podľa článku 8 ods. 1 základného nariadenia.
(94) Tento exportujúci výrobca je výrobca rôznych typov výrobkov, ktoré sa môžu predávať súčasne. Predstavuje to potenciálne riziko krížovej kompenzácie, keď by sa vyrovnávacie ceny formálne rešpektovali, avšak ceny výrobkov iných ako je predmetný výrobok by boli nižšie, ak by sa výrobky predávali súčasne s predmetným výrobkom. To by porušilo záväzok o minimálnej cene kremíka, umožnilo by to jeho obchádzanie a veľmi ťažké by bolo účinne to sledovať.
(95) Vzhľadom na hore uvedené sa dohodlo, že vyrovnanie ponúknuté bezprostredne po ukončení konečných zistení sa nemôže akceptovať v terajšej forme. Záujmové strany boli následne informované a nedostatky ponúkaného vyrovnania boli do detailov vysvetlené exportérom.
PRIJALA TOTO NARIADENIE:
Článok 1
1. Týmto sa na dovozy kremíka s obsahom kremíka nižším ako 99,99 % váhy, patriaceho pod číselný znak KN 28046900 a pochádzajúceho z Ruska, ukladá konečné antidumpingové clo.
2. Sadzba konečného antidumpingového cla uplatneného na výrobok vyrábaný dole uvedenými spoločnosťami s pôvodom v Rusku je takáto:
Spoločnosti | Colná sadzba (%) | Dodatočný kód TARIC |
OJSC Bratsk Aluminium Plant, Bratsk, Irkutský región, Rusko | 23,6 % | A464 |
SKU LLC, Sual - Kremny - Ural, Kamensk, región Ural, Rusko, a ZAO KREMNY, Irkuck, Irkutský región, Rusko | 22,7 % | A465 |
Všetky ostatné spoločnosti | 23,6 % | A999 |
3. Ak nie je uvedené inak, uplatňujú sa platné ustanovenia týkajúce sa colný poplatkov.
Článok 2
Čiastky zabezpečené ako dočasné antidumpingové clá v zmysle nariadenia č. 1235/2003 pre dovoz kremíka, s obsahom kremíka nižším ako 99,99 % váhy, číselného znaku KN 28046900, s pôvodom v Rusku, sa s konečnou platnosťou vyberajú v súlade s uvedenými pravidlami.
Zabezpečené čiastky, ktoré sú vyššie ako sadzby konečného antidumpingového cla, sa uvoľnia. Ak sú konečné clá vyššie ako dočasné clá, s konečnou platnosťou sa vyberajú len čiastky zabezpečené na úrovni dočasných ciel.
Článok 3
Toto nariadenie nadobúda účinnosť dňom nasledujúcim po jeho uverejnení v Úradnom vestníku Európskych spoločenstiev.
Toto nariadenie je záväzné vo svojej celistvosti a priamo uplatniteľné vo všetkých členských štátoch.
V Bruseli 22. decembra 2003""")

```

</div>

## Results

```bash
Aarhus (okres),obchodovanie štátu,veľkoobchod,lepidlo
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legmulticlf_multieurlex_slovak|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|sk|
|Size:|13.0 MB|

## References

https://huggingface.co/datasets/nlpaueb/multi_eurlex

## Benchmarking

```bash
 
labels               precision    recall  f1-score   support
0       0.61      0.42      0.50        90
1       0.91      0.86      0.88      1049
2       0.90      0.92      0.91      1101
3       0.74      0.57      0.65       203
4       0.78      0.84      0.81        45
5       0.87      0.72      0.79        83
6       0.88      0.14      0.25        49
7       0.96      0.77      0.86        31
8       0.82      0.71      0.76       563
9       0.81      0.63      0.71        41
10      0.00      0.00      0.00        11
11      0.95      0.96      0.96       310
12      0.76      0.77      0.77       735
13      0.84      0.94      0.89        17
14      0.77      0.69      0.72       271
15      0.71      0.65      0.68        26
   micro-avg       0.85      0.80      0.83      4625
   macro-avg       0.77      0.66      0.70      4625
weighted-avg       0.85      0.80      0.82      4625
 samples-avg       0.84      0.80      0.80      4625
F1-micro-averaging: 0.8265953892415636
ROC:  0.8863919323198561

```