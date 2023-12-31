# davinci-002-finetuning-bandi-puglia

## davinci-002-Finetuning-Bandi-Puglia - Cartella Documenti

Questa cartella contiene i documenti pdf scaricati dal sito https://www.sistema.puglia.it/ che contiene i bandi della Regione Puglia che ho utilizzato per creare il dataset per il Fine-Tuning


## davinci-002-Finetuning-Bandi-Puglia - Cartella Notebook

Questo progetto contiene alcuni notebook Jupyter¹ che mostrano come utilizzare il modello da-vinci-002 finetunato sui bandi della Regione Puglia² per rispondere a domande specifiche.

### Notebook disponibili

- **01_Dataset_Building.ipynb**: questo notebook mostra come preparare il dataset di domande e risposte sui bandi della Regione Puglia, partendo da PDF³ che contengono i bandi. Il notebook utilizza la libreria **LangChain** per estrarre il testo dal PDF e per splittare il testo.
- **02_FineTuning_e_Costi.ipynb**: questo notebook mostra come effettuare il fine-tuning del modello da-vinci-002 di OpenAI sul dataset creato nel notebook precedente.
- **03_Secondo_FineTuning_e_Costi.ipynb**: questo notebook mostra come effettuare il fine-tuning con 6 epoche del modello da-vinci-002 di OpenAI sul dataset con le sole domande generate e i chunk non migliorati da davinci-003 .
- **04_Fine_Tuning_Evaluations.ipynb**: questo notebook mostra come si comportano entrambi i modelli fine-tuned con diversi prompt. 
- **05_Confronto_Rag.ipynb**: questo notebook mostra come si crea una Retrieval Augmented Generation e mostra le differenze con il modello fine-tuned.
- **Chatbot.py**:  mostra come creare un'applicazione web con la libreria **streamlit** che permette di interagire con il modello finetunato e di fare domande sui bandi della Regione Puglia. L'applicazione mostra anche la fonte della risposta e il link al bando corrispondente.
- **99_QA_with_filtering_Metadata_Label.ipynb**: In questo notebook ho svolto una RAG e ho trovato un modo per creare un metadato label che sia contenuto nel documento, e durante la QA sia possibile rispondere più precisamente filtrando il metadato label trovato anche sulla query.
- **100_Dimensione_impresa_metadata_filtering_RAG** : In questo notebook analizzo solo come etichettare il tipo di impresa a cui è rivolto il documento e fare una retrieval augmentation filtrata in base al tipo di impresa che si da in input.  p.s.(Se ad esempio ho Piccole e Medie Imprese e vado a trovare un filtro Piccole Imprese nei k chunk che trovo potrò trovare anche i chunk etichettati Piccole e Medie Imprese, perchè la label contiente la stringa (Piccole Imprese))  
