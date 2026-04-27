Overview
The goal of this competition is to develop machine learning frameworks capable of identifying understudied species within continuous audio data from Brazil's Pantanal wetlands. Successful solutions will help advance biodiversity monitoring in the last wild places on Earth.

Start

a month ago
Close

2 months to go
Merger & Entry
Description
How do you protect an ecosystem you can’t fully see? One way is to listen.

This competition involves building models that automatically identify wildlife species from their vocalizations in audio recordings collected across the Pantanal wetlands. This work will support more reliable biodiversity monitoring in one of the world’s most diverse and threatened ecosystems.

Understanding how ecological communities respond to environmental change and restoration efforts is a central challenge in conservation science. The Pantanal — a wetland spanning 150,000+ km² across Brazil and neighboring countries — is home to over 650 bird species plus countless other animals, yet much of it remains unmonitored. Seasonal flooding, wildfires, agricultural expansion, and climate change make regular fieldwork challenging.

Photo of a jaguar with its mouth open.

Goal of the Competition
Conventional biodiversity monitoring across vast, remote regions is expensive and logistically demanding. To help address these challenges, a growing network of 1,000 acoustic recorders is being deployed across the Pantanal, running continuously to capture wildlife sounds across different habitats and seasons. Continuous audio recording allows researchers to capture multi-species soundscapes over extended periods, providing a community-level perspective on biodiversity dynamics. But the sheer volume of audio is too large to review manually, and labeled species data is limited.

This competition focuses on the development of machine learning models that identify wildlife species from passive acoustic monitoring (PAM). Proposed approaches should work across different habitats, withstand the constraints of messy, field-collected data, and support evidence-based conservation decisions. Successful solutions will help advance biodiversity monitoring in the last wild places on Earth, including research initiatives in the Pantanal wetlands of Brazil.

Listening carefully, and at scale, may be one of the most effective tools available to protect this landscape.

Timeline
March 11, 2026 - Start Date.

May 27, 2026 - Entry Deadline. You must accept the competition rules before this date to compete.

May 27, 2026 - Team Merger Deadline. This is the last day participants may join or merge teams.

June 3, 2026 - Final Submission Deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

Evaluation
The evaluation metric for this contest is a version of macro-averaged ROC-AUC that skips classes that have no true positive labels.

Submission Format
For each row_id, you should predict the probability that a given species was present. There is one column per species. Each row covers a five-second window of audio.

Prizes
1st Place - $15,000
2nd Place - $10,000
3rd Place - $8,000
4th Place - $7,000
5th Place - $5,000
Best working note award (optional):

Participants of this competition are encouraged to submit working notes to the CLEF 2026 conference. A best BirdCLEF+ working note competition will be held as part of the conference. The top two best working note award winners will receive $2,500 each. See the Evaluation page for judging criteria.

Working Note Award (optional)
Working Note Submission Timeline
We encourage participants to submit a working note write-up of their approach to the CLEF conference. Organizers will award $5,000 in prize money ($2,500 each) for the two best working note submissions.

Submission dates are:

June 3, 2026 - Competition deadline
June 17, 2026 - Working note submission deadline
June 24, 2026 - Notification of acceptance
July 6, 2026 - Camera-ready submission deadline
Additional information on the submission process will be posted ahead of time on the discussion forum.

Working Note Award Criteria
Criteria for the BirdCLEF+ Best Working Note Award:

Originality. The value of a paper is a function of the degree to which it presents new or novel technical material. Does the paper present results previously unknown? Does it push forward the frontiers of knowledge? Does it present new methods for solving old problems or new viewpoints on old problems? Or, on the other hand, is it a rehash of information already known?

Quality. A paper's value is a function of the innate character or degree of excellence of the work described. Was the work performed or the study made with a high degree of thoroughness? Was high engineering skill demonstrated? Is an experiment described which has a high degree of elegance? Or, on the other hand, is the work described pretty much of a run-of-the-mill nature?

Contribution. The value of a paper is a function of the degree to which it represents an overall contribution to the advancement of the art. This is different from originality. A paper may be highly original but may be concerned with a very minor, or even insignificant, matter or problem. On the other hand, a paper may make a great contribution by collecting and analyzing known data and facts and pointing out their significance. Or, a fine exposition of a known but obscure or complex phenomenon or theory or system or operating technique may be a very real contribution to the art. Obviously, a paper may well score highly on both originality and contribution. Perhaps the important question is, will the engineer who reads the paper be able to practice his profession more effectively because of having read it?

Presentation. The value of the paper is a function of the ease with which the reader can determine what the author is trying to present. Regardless of the other criteria, a paper is not good unless the material is presented clearly and effectively. Is the paper well written? Is the meaning of the author clear? Are the tables, charts, and figures clear? Is their meaning readily apparent? Is the information presented in the paper complete? At the same time, is the paper concise?

Evaluation of the submitted BirdCLEF+ working notes:

Each working note will be reviewed by two reviewers and scores averaged. Maximum score: 15.

a) Evaluation of work and contribution

5 points: Excellent work and a major contribution

4 points: Good, solid work of some importance

3 points: Solid work but a marginal contribution

2 points: Marginal work and minor contribution

1 point: Work doesn't meet scientific standards

b) Originality and novelty

5 points Trailblazing

4 points: A pioneering piece of work

3 points: One step ahead of the pack

2 points: Yet another paper about…

1 point: It's been said many times before

c) Readability and organization

5 points: Excellent

4 points: Well written

3 points: Readable

2 points: Needs considerable work

1 point: Work doesn't meet scientific standards

Code Requirements
This is a Code Competition

Submissions to this competition must be made through Notebooks. For the "Submit" button to be active after a commit, the following conditions must be met:

CPU Notebook <= 90 minutes run-time
GPU Notebook submissions are disabled. You can technically submit but will only have 1 minute of runtime.
Internet access disabled
Freely & publicly available external data is allowed, including pre-trained models
Submission file must be named submission.csv
Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you encounter submission errors.

Acknowledgements
The development of the competition dataset was supported by the Bezos Earth Fund AI for Climate and Nature Grand Challenge.

Compiling this extensive dataset was a major undertaking, and we are very thankful to the many domain experts who helped to collect and manually annotate the data for this competition. Specifically, we would like to thank (institutions and individual contributors in alphabetic order):

Chemnitz University of Technology: Stefan Kahl, Mario Lasseck, and Maximilian Eibl

Google Deepmind: Tom Denton

iNaturalist: Grant van Horn

Instituto Homem Pantaneiro: Wener Hugo Arruda Moreno

Instituto Nacional de Pesquisa do Pantanal (INPP): Carolline Zatta Fieker, Karl-L. Schuchmann, Kirk Thiago Pedroso Azevedo, Lucas Korzune Sampaio Teles, Marinez Isaac Marques and Matheus Gonçalves dos Reis

K. Lisa Yang Center for Conservation Bioacoustics: Stefan Kahl, Larissa Sugai and Holger Klinck

LifeCLEF: Alexis Joly and Henning Müller

Sauá Consultoria Ambiental: Carolina Martins Garcia

Universidade Federal de Mato Grosso do Sul (UFMS): Alyson Vieira de Melo, Daiene Louveira Hokama Sousa, José Luiz Massao Moreira Sugai, João Emílio de Almeida Júnior, Liliana Piatti, Mariana Motti Barbosa, Matheus de Oliveira Neves, Priscila do Nascimento Lopes and Ryan Christopher Kridler

Xeno-canto: Willem-Pier Vellinga, Bob Planqué

Photo Credits

Banner picture of a Hyacinth Macaw by Thomas Fuhrmann. Inset picture of a Jaguar by Leonardo Ramos.

Citation
Stefan Kahl, Tom Denton, Larissa Sugai, Liliana Piatti, Ryan Holbrook, Holger Klinck, and Ashley Oldacre. BirdCLEF+ 2026. https://kaggle.com/competitions/birdclef-2026, 2026. Kaggle.

Dataset Description
Your challenge in this competition is to identify which species (birds, amphibians, mammals, reptiles, insects) are calling in recordings made in the Brazilian Pantanal. This is an important task for scientists who monitor animal populations for conservation purposes. More accurate solutions could enable more comprehensive monitoring.

This competition uses a hidden test set. When your submitted notebook is scored, the actual test data will be made available to your notebook.

Files
train_audio/ The training data consists of short recordings of individual bird, amphibian, reptile, mammal, and insect sounds generously uploaded by users of xeno-canto.org and iNaturalist. These files have been resampled to 32 kHz where applicable to match the test set audio and converted to the ogg format. Filenames consist of [collection][file_id_in_collection].ogg. The training data should have nearly all relevant files; we expect there is no benefit to looking for more on xeno-canto.org or iNaturalist and appreciate your cooperation in limiting the burden on their servers. If you do, please make sure to adhere to the scraping rules of these data portals.

test_soundscapes/ When you submit a notebook, the test_soundscapes directory will be populated with approximately 600 recordings to be used for scoring. They are 1 minute long and in ogg audio format, resampled to 32 kHz. The file names have the general form of BC2026_Test_<file ID>_<site>_<date>_<time in UTC>.ogg (e.g., file BC2026_Test_0001_S05_20250227_010002.ogg has file ID 0001, was recorded at site S05 on Feb 27 2025 at 01:00 UTC). It should take your submission notebook approximately five minutes to load all the test soundscapes. Not all species from the training data actually occur in the test data.

train_soundscapes/ Additional audio data from roughly the same recording locations as the test_soundscapes. Filenames follow the same naming convention as the test_soundscapes; although some recording sites overlap between train and test, precise recording dates and times do NOT overlap with recordings of the hidden test data. This year, some of the train_soundscapes have been labeled by expert annotators, and we provide the ground truth for a subset of train_soundscapes in train_soundscapes_labels.csv with columns filename referencing the soundscape file, start and end referencing the 5-second segment for which column primary_label provides a semicolon-separated list of species codes that have been marked as present in this segment.

Important note: Some species with occurrences in the hidden test data might only have train samples in the labeled portion of train_soundscapes and not in the train_audio (XC and iNat data). However, not all species from train_soundscapes have occurrences in the test_soundscapes.

train.csv A wide range of metadata is provided for the training data. The most directly relevant fields are:

primary_label: A code for the species (eBird code for birds, iNaturalist taxon ID for non-birds). You can review detailed information about the species by appending codes to eBird and iNaturalist taxon URL, such as https://ebird.org/species/brnowl for the Barn Owl or https://www.inaturalist.org/taxa/41970 for the Jaguar. Not all species have their own pages; some links might fail.
secondary_labels: List of species labels that have been marked by recordists to also occur in the recording. Can be incomplete.
latitude & longitude: Coordinates for where the recording was taken. Some bird species may have local call 'dialects,' so you may want to seek geographic diversity in your training data.
author: The user who provided the recording. Unknown if no name was provided.
filename: The name of the associated audio file.
rating: Values in 1..5 (1 - low quality, 5 - high quality; 0.5 reduction in rating when background species are present) provided by users of Xeno-canto; 0 implies no rating is available; iNaturalist does not provide quality ratings.
collection: Either XC or iNat, indicating which collection the recording was taken from. Filenames also reference the collection and the ID within that collection.
sample_submission.csv A valid sample submission.

row_id: A slug of [soundscape_filename]_[end_time] for the prediction; e.g., Segment 00:15-00:20 of 1-minute test soundscape BC2026_Test_0001_S05_20250227_010002.ogg has row ID BC2026_Test_0001_S05_20250227_010002_20.
[species_id]: There are 234 species ID columns. You will need to predict the probability of the presence of each species for each row.
taxonomy.csv - Data on the different species, including iNaturalist taxon ID and class name (Aves, Amphibia, Mammalia, Insecta, Reptilia). Most insect species in this competition have not been identified on species level and instead occur as sonotypes (e.g., 47158son16 as insect sonotype 16); these sonotypes are treated as classes despite the lack of species ID and some of them also occur in the test data. The 234 rows of this file represent the 234 class columns in the submission file. primary_label specifies the submission file column name.

recording_location.txt - Some high-level information on the recording location (Pantanal, Brazil).