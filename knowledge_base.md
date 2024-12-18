# Strategia di Investimento Integrata su QuantConnect

## 1. Visione d’Insieme e Obiettivo della Strategia
La strategia mira a costruire un portafoglio azionario solido e dinamico, in grado di adattarsi a differenti regimi di mercato. L’obiettivo è quello di:

- Comprendere il contesto macroeconomico globale.
- Scegliere i settori che meglio si adattano a tale contesto.
- Selezionare i singoli titoli più coerenti.
  
L’utilizzo del machine learning è previsto come supporto, per affinare stime e ridurre il rischio di overfitting, non come base primaria. L’intero approccio sarà sviluppato in Python e compatibile con l’ambiente QuantConnect.

## 2. Analisi Macroeconomica
La prima fase si concentra sulla definizione del regime economico attuale, sfruttando dati macro FRED integrati in QuantConnect.

- **Origine dei Dati**: Indicatori macroeconomici dal database FRED.  
- **Scelta degli Indicatori**: Nessun vincolo predefinito; si selezionano gli indicatori sulla base di conoscenze specialistiche e del contesto.  
- **Logica di Analisi**: Non limitarsi a un singolo indicatore, ma costruire un quadro macro complessivo basato su più parametri (crescita, inflazione, tassi, occupazione, ecc.), valutando la loro interazione.  
- **Output**: Identificare il regime di mercato (es. espansione, contrazione, elevata volatilità) da cui dipenderanno le fasi successive.

## 3. Analisi Settoriale Basata sul Contesto Economico
Comprendere il regime macro serve a individuare i settori potenzialmente favoriti dalle condizioni attuali.

- **Nessun Indicatore Predefinito**: La scelta delle metriche settoriali è libera, guidata dalle competenze dell’analista e dalla coerenza con il contesto macro.  
- **Criteri Possibili**: Solidità finanziaria media del settore, prospettive di crescita aggregate, sensibilità a determinati fattori macro.  
- **Risultato**: Lista ristretta di settori promettenti, coerenti con il regime economico identificato.

## 4. Selezione dei Titoli all’Interno dei Settori Identificati
Dopo aver selezionato i settori, si passa all’analisi dei singoli titoli.

- **Approccio Fondamentale Personalizzato**: Nessun set di metriche fisse; si valutano utili, bilanci, prospettive, qualità del management, leva finanziaria, trend degli utili attesi.  
- **Machine Learning come Supporto**: Il ML entra in gioco solo dopo aver identificato i titoli candidati, per raffinare le previsioni di crescita/rischio e confermare o mettere in dubbio le ipotesi formulate.  
- **Risultato**: Ottenere una lista di titoli che offrano un buon equilibrio fra rendimento atteso e solidità, coerente con il contesto macro e settoriale.

## 5. Predizione dei Ritorni Attesi e Ottimizzazione del Portafoglio
Una volta selezionati i titoli, si crea un portafoglio coerente e resiliente.

- **Allocazione e Fattori di Rischio**: Si tiene conto della dinamica fra i titoli e di come possano reagire a variazioni macro. Non si utilizzano regole rigide, ma si adotta flessibilità in base alle competenze dell’analista.  
- **Obiettivo dell’Ottimizzazione**: Massimizzare il rendimento atteso coerentemente con il regime macro, minimizzare l’esposizione a shock indesiderati, garantire diversificazione.  
- **Nessun Modello Statico**: La scelta dei fattori di rischio, delle metriche di diversificazione, del bilanciamento è lasciata alla sensibilità dello sviluppatore.

## 6. Gestione Dinamica e Monitoraggio Continuo
La strategia non è statica, ma si adatta ai cambiamenti del mercato.

- **Aggiornamenti Periodici**: Rivalutare regolarmente il contesto macro, la selezione settoriale e la lista dei titoli.  
- **Reattività ad Eventi Inattesi**: In caso di shock imprevisti, la composizione del portafoglio deve poter essere modificata rapidamente.  
- **Controllo dei Parametri Chiave**: Monitoraggio costante degli indicatori macro e dei fondamentali dei titoli in portafoglio.

## 7. Implementazione in Ambiente QuantConnect
La strategia verrà implementata in Python, pienamente integrata con QuantConnect.

- **Approccio Tecnico Libero**: Nessun vincolo su librerie, metriche o funzioni specifiche. Lo sviluppatore sceglierà gli strumenti più adatti.  
- **Flessibilità di Sviluppo**: L’architettura è pensata per essere personalizzabile ed evolutiva, consentendo modifiche e aggiornamenti in base all’esperienza acquisita e alla variazione delle condizioni di mercato.

## Risultato Finale
L’approccio delineato mira a:

- Comprendere il contesto macro (attraverso dati FRED scelti ad hoc).
- Identificare settori e titoli coerenti con tale contesto, su basi fondamentali personalizzate.
- Sfruttare il machine learning come supporto, non come fondamento, per evitare overfitting.
- Ottimizzare il portafoglio per massimizzare la resilienza e il rendimento atteso.
- Mantenere una gestione dinamica, aggiornata e pronta a reagire a scenari mutevoli.

Il risultato è una strategia integrata, flessibile e razionale, che lascia allo sviluppatore piena libertà di scegliere i dati e i parametri più adeguati, garantendo una base solida e coerente.

---

## Istruzioni Dirette per l’Implementazione in QuantConnect (Bozza Prompt per GitHub Copilot)

**Nota per Copilot**:  
Tu, all’interno dell’ambiente QuantConnect, devi sviluppare in Python l’intera strategia così come descritta. Hai completa autonomia nella scelta di dati, parametri e metriche, senza ambiguità. L’obiettivo è tradurre in codice questo flusso logico, costruendo un sistema coerente che passi dall’analisi macro, alla scelta dei settori, alla selezione dei titoli, fino all’ottimizzazione e al monitoraggio continuo.  
Dovrai:

1. Accedere ai dati macro dal database FRED in QuantConnect e individuare tu stesso gli indicatori più adeguati.
2. Definire il regime macro (espansione, contrazione, volatilità, ecc.) basandoti su valutazioni autonome dei dati scelti.
3. Individuare i settori coerenti con il regime macro, selezionando in base alla logica economica più opportuna.
4. Analizzare i titoli dei settori scelti su basi fondamentali. Scegli tu le metriche rilevanti (bilanci, utili, prospettive, costi, management, leva finanziaria, ecc.).
5. Usare il machine learning solo dopo aver filtrato i titoli, per rifinire stime e previsioni, non come motore primario.
6. Ottimizzare il portafoglio per equilibrare rischi e rendimenti, senza seguire regole statiche ma sfruttando la tua conoscenza per stabilire la giusta combinazione di asset.
7. Aggiornare periodicamente l’analisi macro, la selezione settoriale, i titoli e la composizione del portafoglio, reagendo prontamente a eventi imprevisti.
8. Sviluppare l’intero codice in Python, integrato con QuantConnect, senza vincoli su librerie o modelli, a patto che l’implementazione sia coerente e completa.

In altre parole, tu devi realizzare un’implementazione pratica e integrata dell’intera logica descritta in questo documento.

## Riferimenti al Codice

### 1. Visione d’Insieme e Obiettivo della Strategia
- [main.py - Initialize()](main.py#L10)

### 2. Analisi Macroeconomica
- [main.py - AnalyzeMacroeconomicData()](main.py#L210)
- [main.py - OnData()](main.py#L110)

### 3. Analisi Settoriale Basata sul Contesto Economico
- [main.py - SelectSectors()](main.py#L240)

### 4. Selezione dei Titoli all’Interno dei Settori Identificati
- [main.py - CoarseSelectionFunction()](main.py#L300)
- [main.py - FineSelectionFunction()](main.py#L310)

### 5. Predizione dei Ritorni Attesi e Ottimizzazione del Portafoglio
- [main.py - ApplyMachineLearning()](main.py#L370)
- [main.py - OptimizePortfolio()](main.py#L400)

### 6. Gestione Dinamica e Monitoraggio Continuo
- [main.py - RebalancePortfolio()](main.py#L490)
- [main.py - TrackPerformance()](main.py#L520)

### 7. Implementazione in Ambiente QuantConnect
- [main.py - Initialize()](main.py#L10)
