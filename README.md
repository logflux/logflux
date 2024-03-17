# LogFlux: A software suite for the automatic log parsing

## Installation

```
pip install .
```
## Functional Tests

```
cd tests
python3 test.py > output.log
```

## Source Overview

* `src/logflux` - The main source code for the LogFlux suite. Each algorithm is implemented in a separate file.
* `tests/test.py` - A set of functional tests for the LogFlux suite.

## Parsers
| Parser     | Venue | Paper|
|:----------:|:-------------:|:---:|
| [ael](https://github.com/logflux/logflux/blob/main/src/logflux/ael.py)        | QSIC'08     | [Abstracting Execution Logs to Execution Events for Enterprise Applications](https://www.scribd.com/document/714227537/AEL-Abstracting-Execution-Logs-to-Execution-Events-for-Enterprise-Applications)|
| [dlog](https://github.com/logflux/logflux/blob/main/src/logflux/dlog.py)       | J. Supercomput.'17     |[Diagnosing Router Events With Syslogs for Anomaly Detection](https://www.scribd.com/document/714227533/Dlog-Diagnosing-Router-Events-With-Syslogs-for-Anomaly-Detection)|
| [drain](https://github.com/logflux/logflux/blob/main/src/logflux/drain.py)      | ICWS'17     |[Drain: An online log parsing approach with fixed depth tree](https://www.scribd.com/document/714227550/Drain-an-Online-Log-Parsing-Approach-With-Fixed-Depth-Tree)|
| [fttree](https://github.com/logflux/logflux/blob/main/src/logflux/fttree.py)     | IWQoS'17|[Syslog Processing for Switch Failure Diagnosis and Prediction in Datacenter Networks](https://www.scribd.com/document/714227535/Fttree-Syslog-Processing-for-Switch-Failure-Diagnosis-and-Prediction-in-Datacenter-Networks)|
| [iplom](https://github.com/logflux/logflux/blob/main/src/logflux/iplom.py)      | KDD'09 |[Clustering Event Logs Using Iterative Partitioning](https://www.scribd.com/document/714227553/IPLOM-Clustering-Event-Logs-Using-Iterative-Partitioning)|
| [lenma](https://github.com/logflux/logflux/blob/main/src/logflux/lenma.py)      | CNSM'15 |[Length Matters: Clustering System Log Messagesusing Length of Words](https://www.scribd.com/document/714227541/Lenma-Length-Matters-Clustering-System-Log-Messages)|
| [lfa](https://github.com/logflux/logflux/blob/main/src/logflux/lfa.py)        | MSR'10 |[Abstracting Log Lines to Log Event Types for Mining Software System Logs](https://www.scribd.com/document/714227538/LFA-Abstracting-Log-Lines-to-Log-Event-Types-for-Mining-Software-System-Logs)|
| [lke](https://github.com/logflux/logflux/blob/main/src/logflux/lke.py)        | ICDM'09 |[Execution Anomaly Detection in Distributed Systems through Unstructured Log Analysis](https://www.scribd.com/document/714031235/Execution-Anomaly-Detection-in-Distributed-Systems-through-Unstructured-Log-Analysis)|
| [logcluster](https://github.com/logflux/logflux/blob/main/src/logflux/logcluster.py) | CNSM'15 | [LogCluster - A Data Clustering and Pattern Mining Algorithm for Event Logs](https://www.scribd.com/document/714227547/LogCluster-a-Data-Clustering-and-Pattern-Mining)|
| [logmine](https://github.com/logflux/logflux/blob/main/src/logflux/logmine.py)    | CIKM'16 |[LogMine: Fast Pattern Recognition for Log Analytics](https://www.scribd.com/document/714227548/LogMine-Fast-Pattern-Recognition-for-Log-Analytics)|
| [logsig](https://github.com/logflux/logflux/blob/main/src/logflux/logsig.py)     | CIKM'11 |[LogSig: Generating System Events from Raw Textual Logs](https://www.scribd.com/document/714227534/LogSig-Generating-System-Events-From-Raw-Textual-Logs)|
| [molfi](https://github.com/logflux/logflux/blob/main/src/logflux/molfi.py)      | ICPC'18 |[A Search-based Approach for Accurate Identification of Log Message Formats](https://www.scribd.com/document/714227542/Molfi-a-Search-based-Approach-for-Accurate-Identification-of-Log-Message-Formats)|
| [nulog](https://github.com/logflux/logflux/blob/main/src/logflux/nulog.py)      | ECML-PKDD'20 |[Self-Supervised Log Parsing](https://www.scribd.com/document/714227536/Nulog-Self-Supervised-Log-Parsing)|
| [shiso](https://github.com/logflux/logflux/blob/main/src/logflux/shiso.py)      | SCC'13 |[Incremental Mining of System Log Format](https://www.scribd.com/document/714227545/Shiso-Incremental-Mining-of-System-Log-Format)|
| [slct](https://github.com/logflux/logflux/blob/main/src/logflux/slct.py)       | IPOM'03 |[A Data Clustering Algorithm for Mining Patterns From Event Logs](https://www.scribd.com/document/714227546/Slct-a-Data-Clustering-Algorithm-for-Mining-Patterns-From-Event-Logs)|
| [spell](https://github.com/logflux/logflux/blob/main/src/logflux/spell.py)     | ICDM'16 |[spell: Streaming Parsing of System Event Logs](https://www.scribd.com/document/714227549/Spell-Streaming-Parsing-of-System-Event-Logs)|
| [va](https://github.com/logflux/logflux/blob/main/src/logflux/va.py)         | IPOM'03 |[ A Data Clustering Algorithm for Mining Patterns From Event Logs ](https://www.scribd.com/document/714227539/Va-a-Data-Clustering-Algorithm-for-Mining-Patterns-From-Event-Logs)|
