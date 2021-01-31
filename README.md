# Face-Mask-Detection

Real-time face mask detection voor Informatica Keuzevak Patroonherkenning en Beeldverwerking.

## Opdracht

De opdracht dat werd gegeven aan ons was om een applicatie te maken die gezichtsmaskers herkent op personen. Naast dit moesten wij ook zelf een aparte functionaliteit verzinnen voor bij het programma zelf. Hiervoor hebben wij een teller gemaakt die telt hoeveel personen er in de frame zelf aanwezig zijn, en hoeveel van die personen wel of niet een gezichtsmasker dragen. Ook wordt de frame opgeslagen als afbeelding wanneer er een persoon gedetecteerd is die geen mondkapje draagt.

![](demo.gif)

### Training

Voor de training hebben wij verscheidene bronnen en artikelen bekeken om te zien welke het beste zou passen bij onze opdracht, zoals welke imagesets en methode het beste zou zijn voor het maken van deze applicatie. Hierbij hebben wij gekozen voor een grote image dataset van mensen met en zonder gezichtsmasker, en hebben wij onze applicate getrained met een convolutional neural network met het deep learning framework Keras voor Google's TensorFlow.

#### Dataset

Wij hebben een bestaande dataset gebruikt met gezichten die wel een gezichtsmasker dragen en gezichten die geen gezichtsmasker dragen<sup>1</sup>. Elk van deze datasets bestaat uit ca. 1000 foto's. In totaal hebben wij met het trainen 1376 afbeeldingen gebruikt. Dit is een goed aantal foto's om als dataset te gebruiken. Uit de trainingsresultaten blijkt dat deze dataset dus ook goede resultaten geeft.

Na het trainen kregen we de volgende resultaten, waar we erg tevreden over zijn:

![Overview](https://github.com/BastiaanJansen/Face-Mask-Detection/blob/main/overview.PNG)

![Plot](https://github.com/BastiaanJansen/Face-Mask-Detection/blob/main/plot.png)

In de plot hierboven is te zien dat er niet veel meer gebeurd na ongeveer 7 epochs. Dus wellicht was 20 epochs een beetje overbodig.

### Real-time tracking

In de applicatie wordt de video van je webcam gebruikt. Voor iedere frame worden er vier stappen uitgevoerd:

1. Het detecteren van gezichten.
2. Het verkrijgen van elk individueel gezicht.
3. Het toepassen van onze gezichtsmasker classifier.
4. Is er geen gezichtsmasker gevonden? Sla de frame op als afbeelding.

Hierdoor wordt er voor elk gezicht goed getracked welk gezicht wel of niet een gezichtsmasker heeft.

## Problemen
Het programma zoekt iedere keer eerst naar gezichten om daarna te kijken of die persoon een gezichtsmasker op heeft. Dit betekent dat wanneer een gezicht niet wordt gedetecteerd, er ook niet wordt gezocht naar een gezichtsmasker. Dit kan problematisch zijn wanneer een persoon wel een gezichtsmasker op heeft, maar doordat het programma geen gezicht kan detecteren omdat het gezichtsmasker een groot deel van het gezicht blokkeert, dus geen persoon kan detecteren.

## Mogelijke verbeteringen

In ons programma zijn er natuurlijk een reeds mogelijke verbeteringen die nog mogelijk toegevoegd zou kunnen worden. Het verwezelijken van deze verbeteringen is helaas niet gelukt met betrekking tot gebrek aan tijd. Hier een lijst van mogelijke verbeteringen:

1) Wij hebben een functie geïmplementeerd dat checkt voor een frame waarin het programma geen gezichtsmasker detecteerd. Hierbij slaat het programma de eerste frame op waarin er op een persoon geen gezichtsmasker wordt gevonden. Het probleem bij deze functionaliteit is dat het programma soms, voor een enkele frame of twee, heel eventjes geen gezichtsmasker detecteerd op een persoon. Hierbij wordt er dus een frame onnodig opgeslagen.

Een manier om dit te voorkomen is om een timer te implementeren. Een timer zou er voor zorgen dat het programma een aantal tijd wacht voordat de frame wordt opgeslagen. Dit zorgt er voor dat het programma alleen een frame zou opslaan wanneer iemand daadwerkelijk geen gezichtsmasker op heeft, omdat het programma dan langer dan een paar frames geen gezichtsmasker op een persoon zou detecteren.

2) Zoals gezegd in het kopje <b>Problemen</b> is een van de problemen van het programma dat het soms niet goed gezichten kan detecteren gezien het feit dat de gezichtsmaskers grote delen van een gezicht kan blokkeren. Een mogelijke oplossing zou kunnen zijn om extra detectiemethodes toe te voegen zoals d.m.v. extra feature-detectie. Vooralsnog komt dit probleem niet vaak voor als de gezichtsmasker zelf niet een te groot gebied van een gezicht bedekt.

## Bronnen
- https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/
- https://towardsdatascience.com/covid-19-face-mask-detection-using-tensorflow-and-opencv-702dd833515b
- [1] https://github.com/balajisrinivas/Face-Mask-Detection/tree/master/dataset
