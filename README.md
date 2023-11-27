# davinci-002-finetuning-bandi-puglia

## davinci-002-Finetuning-Bandi-Puglia - Cartella Documenti

Questa cartella contiene i documenti pdf scaricati dal sito www.sistema-puglia.it che contiene i bandi della Regione Puglia che utilizzeremo per creare il dataset per il Fine-Tuning


## davinci-002-Finetuning-Bandi-Puglia - Cartella Notebook

Questo progetto contiene alcuni notebook Jupyter¹ che mostrano come utilizzare il modello da-vinci-002 finetunato sui bandi della Regione Puglia² per rispondere a domande specifiche.

### Notebook disponibili

- **01_Dataset_Building.ipynb**: questo notebook mostra come preparare il dataset di domande e risposte sui bandi della Regione Puglia, partendo da PDF³ che contengono i bandi. Il notebook utilizza la libreria **LangChain** per estrarre il testo dal PDF e per splittare il testo.
- **02_FineTuning_e_Costi.ipynb**: questo notebook mostra come effettuare il fine-tuning del modello da-vinci-002 di OpenAI sul dataset creato nel notebook precedente.
- **03_Secondo_FineTuning_e_Costi.ipynb**: questo notebook mostra come effettuare il fine-tuning con 6 epoche del modello da-vinci-002 di OpenAI sul dataset con le sole domande generate e i chunk non migliorati da davinci-003 .
- **04_Fine_Tuning_Evaluations.ipynb**: questo notebook mostra come si comportano entrambi i modelli fine-tuned con diversi prompt. 
- **05_Confronto_Rag.ipynb**: questo notebook mostra come si crea una Retrieval Augmented Generation e mostra le differenze con il modello fine-tuned.
- **Application.py**:  mostra come creare un'applicazione web con la libreria **streamlit** che permette di interagire con il modello finetunato e di fare domande sui bandi della Regione Puglia. L'applicazione mostra anche la fonte della risposta e il link al bando corrispondente.

