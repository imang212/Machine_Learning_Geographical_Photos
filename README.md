# Trénovací model na geografická data
Rozhodl jsem se udělat trénovací model používající strojové učení na rozpoznávání geografických dat v rámci seminární práce na USU. 

### Dataset
Budu pracovat z datasetem ze stránek kaggle.com (planets_dataset), který obsahuje přibližně 40 000 testovacích, 40 000 trénovacích obrázků terénu amazonského pralesa a obsahuje soubory csv formátu s popisem testovacích a trénovacích obrázků. Datasetem geografických obrázku vyfocených v Amazonském pralese, který obsahuje 2 csv soubory s tabulkami, které popisují, co na obrázcích je pro strojové učení. Dataset můžu volně použít ke strojovému učení na základě license Database Contents License, která to povoluje. Odkaz k datasetu: https://www.kaggle.com/datasets/nikitarom/planets-dataset/data

Pojďme si zobrazit kořenovou strukturu datasetu.:

![image](https://github.com/user-attachments/assets/1fab1cc9-de39-4a88-a92f-6ae9f64b1def)

První csv tabulka s názvem **sample_submission.csv** obsahuje 61191 hodnot, obsahuje 2 sloupce:
image_name - název daného obrázku
tags - popis vlastností toho co je na obrázcích.

Druhá csv tabulka s názvem **train_classes.csv** obsahuje 40 479 hodnot obsahuje také 2 sloupce:
image_name - název daného obrázku
tags - popis vlastností toho co je na obrázcích např.: clear_primary, clear_cloudy_primary, atd...

Máme adresáře pro obrázky:

**test-jpg**, kde se nachází testovací obrázky, je jich přibližně 40 000. 

**train-jpg**, kde se nachází trénovací obrázky, je jich přibližně 40 000.

Máme ještě adresář **test-jpg-additional**, kde se nachází testovací ještě nachází přibližně 20 500 testovacích obrázků navíc. 

### Technologie
Python 3.11. PyTorch a Resnet50.
### Načtení dat pro trénovací model

```python
#import potřebných datasetů
import os
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, multilabel_confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Nastavení cest k datům
DATA_DIR = './nikitarom/planets-dataset/versions/3/'
TRAIN_DIR = os.path.join(DATA_DIR, 'planet/planet/train-jpg')
TEST_DIR = os.path.join(DATA_DIR, 'planet/planet/test-jpg')
TRAIN_CLASSES = os.path.join(DATA_DIR, 'planet/planet/train_classes.csv')
SUBMISSION = os.path.join(DATA_DIR, 'planet/planet/sample_submission.csv')

# Načtení CSV souborů
train_df = pd.read_csv(TRAIN_CLASSES)
submission_df = pd.read_csv(SUBMISSION)
```
Naimportoval jsem potřebné datasety pro práci s daty. Určil jsem si cesty k adresářům s trénovací a testovacími obrázky a s popisem obrázků. Načetl jsem si csv soubory.

### Zjišťovaní informací o tabulce
```python
print(f"Počet trénovacích záznamů: {len(train_df)}")
print(f"Počet testovacích záznamů: {len(submission_df)}")

# Zobrazení informací o tabulce
print('\nZobrazení informací o trénovací tabulce:')
print("Hlavička tabulky: ", train_df.head(), '\n')
print("Informace o tabulce: ", train_df.info(), '\n')
print("Nulové hodnoty: ", train_df.isnull().sum(), '\n')

# Zobrazme si distribuci tagů ve sloupci 'tags'
all_tags = []
for tags in train_df['tags'].values:
    all_tags.extend(tags.split())
unique_tags = sorted(list(set(all_tags)))
print(f"\nPočet unikátních tagů: {len(unique_tags)}")
print(f"Unikátní tagy: {unique_tags}")
```
![image](https://github.com/user-attachments/assets/b7fbb05d-95e6-4d8a-a7f0-8462262d6b22)

Vypsal jsem si informace o tabulce, abych věděl kolik s ní je záznamů, jak má nastavené sloupce, jestli tam například jsou povoleny nulové hodnoty a jakého typu jsou. Také jsem si vypsal informace o tom, jestli obsahuje nějaké nulové hodnoty a kolik je tagů ve sloupci **tags**, kde jsem zjistil, že se vyskytuje 17 různých tagů pro popis prostředí obrázků. 

**Unikátní tagy:** agriculture, artisinal_mine, bare_ground, blooming, blow_down, clear, cloudy, conventional_mine, cultivation, habitation, haze, partly_cloudy, primary, road, selective_logging, slash_burn, water

#### Analýza frekvence tagů
Abych měl přehled o tom kolikrát se tam který tag vyskytuje.
```python
tag_counts = {}
for tag in all_tags:
    if tag in tag_counts: tag_counts[tag] += 1
    else: tag_counts[tag] = 1

sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
print("\nNejčastější tagy:")
for tag, count in sorted_tags[:10]:
    print(f"{tag}: {count} výskytů")
```
![image](https://github.com/user-attachments/assets/cce47905-bde1-4714-a80e-32a0beb21b59)
#### Graf distribuce jednotlivých tagů
![tags_distribution](https://github.com/user-attachments/assets/652ff75c-a222-46e7-a43c-29773f624a26)

### Trénování dat a vytvoření klasifikátorů
#### Vytvoření one-hot encodingu pro tagy
Abych mohl s tagy u obrázků trénovacím modelu pracovat, tak si musím data převét na vektory, abych s nimi mohl pracovat v PyTorch, kde je poté pomocí tensoru převeden na formát vektorů ve floatech, s kterým bude pracovat trénovací model. Vypíšem si hlavičku datasetu a je tam vidět nový vytvořený sloupec, kde u každého záznamu je udělaný vektor pole, který má 17 hodnot ve formátu floatu, které buďto reprezentují 1.0(True) a nebo 0.0(False) hodnotu, podle toho jestli dané tagy obrázek obsahuje.   
```python
# Vytvoření one-hot encodingu pro tagy
def get_tag_map(tags):
    labels = np.zeros(len(unique_tags))
    if pd.isna(tags): return labels
    for tag in tags.split():
        if tag in unique_tags:
            labels[unique_tags.index(tag)] = 1
    return labels
# Přidání one-hot encodingu do dataframe
train_df['tag_vector'] = train_df['tags'].apply(get_tag_map)

print(train_df.head())
```
![image](https://github.com/user-attachments/assets/3c2fc5bc-d3e0-4f06-ae01-ac8fb5161937)

#### Vytvoření dataset třídy pro PyTorch 
Abych mohl udělat trénovací model pro Resnet50, tak si musím vytvořit dataset třídu pro PyTorch. V této třídě si určím dataframe, s kterým budu pracovat, adresář pro obrázky a typ transformace pro dané obrázky. Je tu funkce, která mi vrací délku dataframu a funkce __getitem__, která načte obrázky, aplikuje na ně transformaci a pomocí FloatTensor v PyTorch převede tag_vector sloupec s vektory, který převede vektory tagů ve floatu do správného formátu k trénování datasetu a potom vrátíme vytvořený dataset pro zpracování s obrázky a tag vektorem.
```python
# Vytvoření vlastní Dataset třídy pro PyTorch
class PlanetDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['image_name']
        img_path = os.path.join(self.img_dir, img_name + '.jpg')
        # Načtení obrázku
        image = Image.open(img_path).convert('RGB')
        # Aplikace transformací, pokud jsou definované
        if self.transform: image = self.transform(image)
        # Získání one-hot encodingu pro tagy
        tag_vector = torch.FloatTensor(self.dataframe.iloc[idx]['tag_vector'])
        return image, tag_vector
```
#### Definování transformace
Nadefinujem si transformaci pro trénovací a validační data, která bude použita pro trénování modelu. Můžem si nadefinovat zvlášť transformaci pro trénovací data a pro validační data.

U transformace si můžeme určit.:
- U jaké velikosti v pixelech chceme začít.
- Jestli je převedeme na tensor.
- Jestli chceme obrázek náhodně otáčet horizontálně nebo vertikálně. Tohle se hodně hodí pro obrázky tohoto typu, protože dané obrázky mohou být jakkoliv vyfoceny. 
- Náhodnou rotaci, u které si určíme počet stupňů.
- Můžem si upravit obraz, jaký chceme mít jas, kontrast, sytost, a další...
- Normalizaci, kde určíme průměr a směrodatnou odchylku na třídimenzionální data transformovaní obrázků.
- A spoustu dalších vlastností...
```python
# Definování transformací pro obrázky
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```
#### Rozdělení na trénovací data a jejich připracení pro trénování modelu
Model si rozdělím na trénovací a validační data. Nastavím si rozdělení trénovacích dat a validačních dat 80/20 a random_state pro zamíchání dat. Následně si vypíši kolik tam je trénovacích a validačních vzorků Vytvořím si datasety pro trénovací a validační data podle třídy PlanetDataset definované pro train model pomocí torch. 
```python
train_data, valid_data = train_test_split(train_df, test_size=0.2, random_state=42)
print(f"\nPočet trénovacích vzorků: {len(train_data)}")
print(f"Počet validačních vzorků: {len(valid_data)}")
# Vytvoření datasetů
train_dataset = PlanetDataset(train_data, TRAIN_DIR, transform=train_transform)
valid_dataset = PlanetDataset(valid_data, TRAIN_DIR, transform=val_transform)
```
![image](https://github.com/user-attachments/assets/68e96cc7-eec2-4446-85d6-f15e2d43b884)

Načtení dat do DataLoaderu a vytvoření vizualizace.

Datasety si načtu do DataLoaderu obsahující dávky (iterace) dat pomocí funkce z PyTorch, abych z nich mohl udělat trénovací model, ve kterých si určím batch size, který obvykle bývá 32, jestli je chci zamíchat a počet workerů. A potom si můžu vypsat ukázku načtení jedné dávky dat obrázků a labelů z train DataLoaderu. Tato operace, pokud těch dat je hodně, může už trvat dlouhou dobu. Můžu si vypsat, jak vypadá taková jedna dávka dat připravená pro vytvoření trénovacího modelu pomocí resnet50. Vizualizuji si obrázky z datasetu. Na konci si ještě můžu nastavit hodnoty deformací obrázků pro průměr a smerodatnou odchylku, poté už jsou data připravena pro trénování.
```python
# Vytvoření dataloaderů
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# Ukázka načtení jedné dávky dat
start_time = time.time()
images, labels = next(iter(train_loader))
end_time = time.time()
print(f"\nTvar načtených obrázků: {images.shape}")
print(f"Tvar načtených tagů: {labels.shape}")
print("Čas pro načtení dat do DataLoaderu: ", end_time - start_time, " s")
# Vizualizace několika obrázků z datasetu
mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
def visualize_sample(dataset, num_samples=5):
    plt.figure(figsize=(15, 3*num_samples))
    for i in range(num_samples):
        image, label = dataset[i]
        image = image.permute(1, 2, 0).numpy()
        # Denormalizace obrázku pro zobrazení
        image = std * image + mean
        image = np.clip(image, 0, 1)
        tags = [unique_tags[j] for j in range(len(unique_tags)) if label[j] == 1]
        plt.subplot(1, num_samples, i+1)
        plt.imshow(image)
        plt.title(f"Tagy: {', '.join(tags)}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
visualize_sample(train_dataset)
print("\nData jsou připravena pro trénování modelu!")
```
#### Použití předtrénovaného modelu
Vytvoříme si třídu multi-label klisifikátoru pro klasifikaci daných dat, aby se mohla trénovat. Když jsou data připravená, můžu už začít s předtrénovaným modelem resnet50. Zmrazí se prvních 10 vrstev. Nahradí finální plně připojenou vrstvu pro multi-label klasifikátor. Nastavíme si sekvenci, v jaký sekvenci chceme, aby nám trénovací model trénoval data. nadefinujem si nejdřív lineární transformaci pro plně připojenou vrstvu, poté ReLU, jaký chceme Dropout, poté tohleto ještě můžeme opakovat, jenom lineární transformace bude pro počet tříd. ještě máme udělanou funkci pro vrácení přední vrstvy klasifikátoru.
```python
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# vytvoření multi-label klasifikátoru pomocí ResNet50
class PlanetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PlanetClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # Zmrazení prvních 10 vrstev modelu
        for param in list(self.resnet.parameters())[:-10]:
            param.requires_grad = False
        # Nahrazení finální plně připojené vrstvy pro multi-label klasifikaci
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.resnet(x)
```

#### Vytvoření trénovacího modelu
Vytvoříme si funkci pro trénování modelu, kam nám bude vstupovat náš vytvořený klasifikátor, dataset loader trénovacích dat a dataset loader validačních dat, criterion zaznamenávající ztrátu, optimizer (nejčastější je Adam), scheduler (zaznamenává průběžný learning rate) a epochs (kolik chci provést epoch).

Nejdříve si ve funkci určím device zařízení pro trénování modelu podle toho jestli chci cude (grafickou kartu od Nvidie) a jestli počítač nemá grafiku od Nvidie, tak se může zvolit cpu (procesor).

Teďka je potřeba dosadit do našeho modelu toto zařízení, pomocí kterého se bude učit. Určíme si proměnnou pro zaznamenávání nejlepší hodnoty, podle které si potom na konci uložíme ten nejlepší model. 

Vytvoříme si seznamy pro ztráty tréningových dat a validačních dat, podle kterých poté budeme zobrazovat úspěšnost modelu.

Model budeme učit v epochách (cyklech) tím, že si vytvoříme hlacní for-cyklus pro daný počet epoch, kolik chceme provést. Před dalším for cyklem vypíšeme na kolikáté epoše náš model je.

**1. Trénování dat**

Násleďně pomocí příkazu model.train() začne trénování v dané fázi modelu. Při tomto trénování si budem do proměnné running_loss zaznamenávat ztrátu. For-cyklem projdeme inputy a labely pro loader dataset trénovacích dat, kde je přidáme do paměti grafické karty nebo procesoru, který je bude učit. U optimizéru si nastavíme zero_grad(), nulový gradient. Inputy v modelu převedeme na outputy a násleďně vytvoříme proměnnou loss, kam uložíme podle kriterion, kam dáme outputy s labely. Uděláme loss.backward() ztrátových dat. Dále optimizer.step(). Poté uložíme do proměnné running_loss ztrátový item vynásobený velikostí inputu.

Potom, když nám tento for-cyklus pro danou ecpochu proběhl můžeme udělat proměnnou epoch_train_loss, do které se uloží procentuální ztráta u každé epochy a tu následně uložíme do train_loses seznamu.

**2.Validace dat**

Teď je potřeba ještě projít validační data v dané fázi, kde začne vyhodnocování modelu pomocí příkazu model.eval(), u kterého zase budeme zaznamenávat running_loss ztrátu a bude mít seznamy pro ukládání predikátů a labelů. Pomocí PyTorch s vypnutím výpočtu gradientů s for-cyklem, který má inputy a labely pro loader dataset validačních dat provedeme validační fázi, která už je o něco jednodušší než u trénovacích dat. Zase inputy a labely přidáme do daného zařízení, uděláme outputy a vytvoříme loss proměnnou pomocí criterion, running_loss proměnnou, kde se zaznamená aktuální ztráta při validaci. u validačních dat vytvoříme binární predikce pomocí sigmoidy, kam budou vstupovat outputy a nastaví se u toho minimální dolní mez, odkud chceme, aby nám to bralo. Binární predikce ještě převedeme do float formátu a následně je uložíme do seznamu všech predikcí, které nám vrátí z CPU paměti. Do seznamu všech labelů také uložíme labely, které nám také vrátí z CPU paměti.

Predikáty a labely z validačních dat převedeme do 1D dimenze a do formátu Numpy array. Do proměnné epoch_val_loss si uložíme úspěšnost validace.

**Vyhodnocení dané fáze**

Teď, když proběhlo trénování a validace, uděláme nějaké vyhodnocení dané fáze trénovacího modelu.

Z daných dat uděláme precision score a recall score. Vypočítáme F1 score z dané fáze a potom celkové. Násleďně tyto data vypíšeme. Vypíšeme i F1 score pro nejlepší a nejhorší tagy.

Aktualizujem learning rate pomocí schenduler. Nejlepší model uložíme do souboru typu pth. (Důležité)

**Visualizace výsledků trénovacího modelu**

Nakonec si uděláme graf vizualizace výsledků trénovacího modelu, který si uložíme do souboru a z celé funkce vrátíme model.
```python
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Používám zařízení: {device}")
    model = model.to(device)
    best_val_f1 = 0.0
    train_losses = []; val_losses = []
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        # Validation phase
        model.eval()
        running_loss = 0.0
        all_preds = []; all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                # Convert probabilities to binary predictions using 0.5 threshold
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        # Spojení všech dávek
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        precision = precision_score(all_labels, all_preds, average='samples', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='samples', zero_division=0)
        # F1 skóre pro každou třídu a průměr
        sample_f1 = f1_score(all_labels, all_preds, average='samples', zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        print(f'Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}')
        print(f'Sample F1: {sample_f1:.4f}, Macro F1: {macro_f1:.4f}')
        # Výpis F1 skóre pro některé důležité tagy
        tag_f1_scores = []
        for i, tag in enumerate(unique_tags):
            tag_f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
            tag_f1_scores.append((tag, tag_f1))
        # Seřazení tagů podle F1 skóre
        tag_f1_scores.sort(key=lambda x: x[1], reverse=True)
        print("\nF1 skóre pro nejlepší tagy:")
        for tag, f1 in tag_f1_scores[:5]:
            print(f"{tag}: {f1:.4f}")
        print("\nF1 skóre pro nejhorší tagy:")
        for tag, f1 in tag_f1_scores[-5:]:
            print(f"{tag}: {f1:.4f}")
        # Aktualizace learning rate
        scheduler.step(epoch_val_loss)        
        # Uložení nejlepšího modelu
        if sample_f1 > best_val_f1:
            best_val_f1 = sample_f1
            torch.save(model.state_dict(), 'best_planet_classifier.pth')
            print("Uložen nejlepší model!")
    # Vizualizace výsledků tréninku
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, 'b-', label='Trénovací ztráta')
    plt.plot(range(1, num_epochs+1), val_losses, 'r-', label='Validační ztráta')
    plt.xlabel('Epocha')
    plt.ylabel('Ztráta')
    plt.title('Trénovací a validační ztráta')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_results.png')
    return model
```
### Trénování modelu
Nyní provedeme trénování modelu. Uděláme model z vytvořené vlastní třídy klasifikátoru. Určíme si criterion pro ztrátová data, určíme si optimizer a schenduler. Násleně všechny tyto proměnné dáme do funkce train model, která ám ho vytrénuje. Musí tam vstupovat klasifikátor modelu, daný loader trénovacích dat, loader validačních dat, criterion pro ztrátová data, optimizer, scheduler a počet fází (epoch) v kolika to chceme udělat.

```python
# Inicializace modelu, kritéria, optimizéru a scheduleru
model = PlanetClassifier(num_classes=len(unique_tags))
criterion = nn.BCEWithLogitsLoss()  # Vhodné pro multi-label klasifikaci
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

print("Trénování modelu...")
num_epochs = 15
start_time = time.time()
trained_model = train_model(model=model, train_loader=train_loader, val_loader=valid_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, num_epochs=num_epochs)
end_time = time.time()
print(f"Čas pro vytrénování modelu v {num_epochs} epochách: ", end_time - start_time)
```
### Vyhodnocení nejlepšího modelu
Načteme si ten nejlepší model.
```python
model.load_state_dict(torch.load('best_planet_classifier.pth'))
```
#### Testování modelu na testovacíhch datech
Funkce pro predikci a vytvoření submission souboru pro testovací data na nejlepším modelu.

```python
def create_submission(model, test_loader, submission_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    predictions = []; image_names = []
    with torch.no_grad():
        for batch, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float().cpu().numpy()
            for i in range(len(preds)):
                idx = batch * test_loader.batch_size + i
                if idx < len(submission_df):
                    img_name = submission_df.iloc[idx]['image_name']
                    image_names.append(img_name)
                    # Získání tagů pro predikci
                    pred_tags = []
                    for j, val in enumerate(preds[i]):
                        if val == 1:
                            pred_tags.append(unique_tags[j])
                    predictions.append(' '.join(pred_tags))  
    # Vytvoření submission dataframe
    submit_df = pd.DataFrame({'image_name': image_names, 'tags': predictions })
    # Uložení do CSV
    submit_df.to_csv('submission.csv', index=False)
    print("Submission soubor byl vytvořen!")

submission_df['tag_vector'] = [np.zeros(len(unique_tags)) for _ in range(len(submission_df))]

test_dataset = PlanetDataset(submission_df, TEST_DIR, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

start_time = time.time()
create_submission(model, test_loader, submission_df)
end_time = time.time()
print(f"Čas pro vytvoření submission pro nejlepší model.: ", end_time - start_time)
```
#### Confusion matrices
Vyhodnocení nejlepšího modelu na testovacích datech pomocí confusion matrices.

```python
def evaluate_model_with_confusion_matrix(model, data_loader, unique_tags):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    all_preds = []; all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    # Spojení všech dávek
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    # 1. Binary Confusion Matrix pro každou třídu
    binary_cm = multilabel_confusion_matrix(all_labels, all_preds)
    # Vizualizace pro vybrané třídy (např. prvních 5 tagů)
    n_classes_to_show = min(5, len(unique_tags))
    plt.figure(figsize=(15, 4 * n_classes_to_show))
    for i in range(n_classes_to_show):
        plt.subplot(n_classes_to_show, 1, i + 1)
        cm = binary_cm[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title(f'Confusion Matrix pro třídu: {unique_tags[i]}'); plt.ylabel('Skutečné hodnoty'); plt.xlabel('Predikované hodnoty')
    plt.tight_layout(); plt.savefig('binary_confusion_matrices.png'); plt.close()
    # 2. Komplexnější přístup - počty správně a špatně predikovaných tagů
    tps = np.sum(np.logical_and(all_preds == 1, all_labels == 1), axis=1)
    fps = np.sum(np.logical_and(all_preds == 1, all_labels == 0), axis=1)
    fns = np.sum(np.logical_and(all_preds == 0, all_labels == 1), axis=1)
    tns = np.sum(np.logical_and(all_preds == 0, all_labels == 0), axis=1)
    # Pro každý vzorek máme nyní čtyři hodnoty - můžeme spočítat průměry
    avg_tp = np.mean(tps)
    avg_fp = np.mean(fps)
    avg_fn = np.mean(fns)
    avg_tn = np.mean(tns)
    # Vytvoření "průměrné" confusion matrix
    avg_cm = np.array([[avg_tn, avg_fp], [avg_fn, avg_tp]])
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title('Průměrná Confusion Matrix (na vzorek)'); plt.ylabel('Skutečné hodnoty'); plt.xlabel('Predikované hodnoty')
    plt.tight_layout(); plt.savefig('avg_confusion_matrix.png'); plt.close()
    # 3. Analýza pro každý tag - výpočet accuracy, precision, recall a F1 score
    tag_metrics = []
    for i, tag in enumerate(unique_tags):
        cm = binary_cm[i]
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        tag_metrics.append({ 'tag': tag, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'support': tp + fn  # Počet výskytů tagu v datasetu
        })
    # Seřazení podle F1 score
    tag_metrics.sort(key=lambda x: x['f1'], reverse=True)
    # 4. Vizualizace metrik pro všechny tagy
    # Získání metrik pro všechny tagy a jejich vizualizace
    tags = [m['tag'] for m in tag_metrics]
    f1_scores = [m['f1'] for m in tag_metrics]
    precisions = [m['precision'] for m in tag_metrics]
    recalls = [m['recall'] for m in tag_metrics]
    supports = [m['support'] for m in tag_metrics]
    # Vizualizace F1 score pro všechny tagy
    plt.figure(figsize=(12, len(tags) * 0.3))
    plt.barh(tags, f1_scores)
    plt.xlabel('F1 Score'); plt.ylabel('Tag'); plt.title('F1 Score pro všechny tagy')
    plt.tight_layout(); plt.savefig('tag_f1_scores.png'); plt.close()
    return binary_cm, avg_cm, tag_metrics
binary_cm, avg_cm, tag_metrics = evaluate_model_with_confusion_matrix(model, test_loader, unique_tags)
```

