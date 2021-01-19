# Face-Mask-Detection

Real-time face mask detection voor Informatica Keuzevak Patroonherkenning en Beeldverwerking.

## Opdracht

De opdracht dat werd gegeven aan ons was om een applicatie te maken die gezichtsmaskers herkent op personen. Naast dit moesten wij ook zelf een aparte functionaliteit verzinnen voor bij het programma zelf. Hiervoor hebben wij een teller gemaakt die telt hoeveel personen er in de frame zelf aanwezig zijn, en hoeveel van die personen wel of niet een gezichtsmasker dragen.

### Training

Voor de training hebben wij verscheidene bronnen en artikelen bekeken om te zien welke het beste zou passen bij onze opdracht, zoals welke imagesets en methode het beste zou zijn voor het maken van deze applicatie. Hierbij hebben wij gekozen voor een grote image dataset van mensen met en zonder gezichtsmasker, en hebben wij onze applicate getrained met een convolutional neural network met het deep learning framework Keras voor Google's TensorFlow.

### Real-time tracking

In de applicatie wordt de video van je webcam gebruikt. Voor iedere frame worden er drie stappen uitgevoerd:

1. Het detecteren van gezichten.
2. Het verkrijgen van elk individueel gezicht.
3. Het toepassen van onze gezichtsmasker classifier.

Hierdoor wordt er voor elk gezicht goed getracked welk gezicht wel of niet een gezichtsmasker heeft.

## Bronnen
- https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/
