# AtmoSound: An ML Framework for Venue-Adaptive Playlist Generation

# [1] Abstract

*In the hyper-competitive restaurant industry, customer experience is the difference between surviving and determines success. Beyond the menu, a restaurant's environment turns new visitors into regulars. Our solution automates this by optimizing music and atmosphere to drive loyalty. We will design and implement a machine learning pipeline that uses feature-level data from Google Maps' Places API such as category tags and review text to generate playlists whose acoustic properties match a venue’s brand identity. The final set of input features will be determined during research.*

*Our team will implement Ridge Regression, Neural Networks, and potentially K-Nearest Neighbours to ingest venue data and accurately create playlists. We will evaluate performance using mean-squared error (MSE), cosine similarity, and top-K genre retrieval accuracy. We expect venue feature-level data to be a strong predictor of desired acoustic features and that our ML methods and UI will outperform human-curated playlists in quality and speed. We also aim to clarify how recommendation algorithms shape human experience and identify which venue characteristics most affect customer experience.*

# [2] Introduction -> - Introduction: There is no overview of existing work on the problem (-1 point) or summary prior efforts to solve the problem and research gaps  (-1 point)

Music is a measurable driver of how customers feel, how long they stay, and how they perceive a brand. Yet most venues still select music manually or rely on existing playlists. Existing recommendation systems are built to optimize for individual listening history, not the atmospheric identity of a physical location. Meanwhile, venues already generate rich public signals through platforms like Google Maps: venue type, customer reviews, category metadata, and foot traffic patterns and attribute flags that remain largely untapped for this purpose. 

This project builds a machine learning pipeline that takes venue-level Google Maps data and generates playlists aligned to a venue's brand identity and atmosphere.   

The pipeline uses two core ML methods: Ridge Regression and a two-layer Neural Network, both trained to predict a multi-dimensional audio profile for a given venue. Training labels are constructed by mapping venue categories and review keywords to representative genres and using their average audio features as targets. Songs are then retrieved by matching the predicted profile against the Spotify tracks dataset, producing a final playlist of around 20-30 tracks. Performance is evaluated using Mean Squared Error (MSE), Cosine Similarity between predicted and target audio profiles, and Top-K Genre Retrieval Accuracy.  

The system is deployed as a Streamlit web application targeting venue owners and brand managers. Users input a Google Maps link and receive a curated playlist tailored to their space. We expect the system to recommend playlists that better match a venue's atmosphere by leveraging the full combination of venue features. We believe this will help venue owners strengthen the overall customer experience and reinforce their brand identity, without needing any music expertise.

# [3] **Background**

Background music directly affects customer sentiment and dwell time, and the effect is strongest when the music aligns with the atmosphere of the space\~\\cite{Roschk2017Calibrating, Wirtz2001Congruency}. However, research on atmospheric cues shows that venues often rely on heuristics and intuition when selecting music and other ambience elements\~\\cite{Douce2022Crossmodal}.

The relationship between music and consumer behavior has been studied since the 1980s. Milliman\~\\cite{Milliman1982Using} demonstrated that slower-tempo music led customers to spend more time and money in retail environments.

Music recommendation systems have evolved considerably, progressing from early collaborative filtering approaches toward content-aware models that incorporate audio signals directly. Van den Oord et al.\~\\cite{VanDenOord2013Deep} demonstrated that CNNs trained on raw audio could recommend songs without prior listening data, and later hybrid models extended this further. However, most existing systems ignore contextual factors such as location or venue type, despite evidence that these conditions strongly shape music preferences. Context-aware music recommendation remains an underexplored area for more systematic approaches\~\\cite{Wang2014Improving, Murciego2021Context, Kaminskas2013Location, Wang2020CAME}.

This project addresses that gap. We use venue-level signals to predict what kind of music fits a given space and retrieve songs accordingly. Specifically, we use publicly available Google Maps data---venue category, attributes, and customer reviews---to construct a feature representation of each venue and map that representation directly to audio features to generate a suitable playlist. We expect this venue-grounded approach to produce better atmospheric fit than genre-based selection, as it draws on richer and more specific signals about each venue.

# [4] End-to-End ML Pipeline

## [4.1] Offline Model Training and Evaluation

All offline development will be done in Jupyter Notebooks. We will iterate on data processing, model training, and evaluation within notebooks before porting final code to the Streamlit app.
### [4.1.1] Data Collection, Exploration & Processing

#### Datasets

We use two datasets:  
**Dataset 1: Manhattan Venues (Google Maps Places API)**
* **Source:** Collected directly via the Google Maps Places API in April 2026\.  
* **Size:** 4,484 venues across Manhattan, 41 columns.  
* **Data types:** Numerical (rating, latitude, longitude), categorical (primary\_type with values like "italian\_restaurant" or "coffee\_shop", price\_level, neighbourhood), boolean (18 attribute flags such as live\_music, outdoor\_seating, serves\_cocktails, allows\_dogs), and free text (generative\_summary and review\_summary ; short paragraph descriptions of each venue generated by Google and by aggregated user reviews).  
* **How we use it:** This is the input side of our pipeline. Each venue gets converted into a feature vector that describes its atmosphere, and our model predicts what kind of music fits that atmosphere.

**Dataset 2: Spotify Tracks** ([https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/data](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/data))

* **Source:** Kaggle (Spotify Tracks Dataset; The data was collected and cleaned using Spotify's Web API and Python).  
* **Size:** 114,000 tracks, 20 columns, spanning 114 genres with exactly 1,000 tracks per genre.  
* **Data types:** Numerical audio features (danceability, energy, acousticness, valence, tempo, loudness, instrumentalness, liveness, speechiness which are all floats between 0 and 1 except tempo and loudness), categorical (track\_genre), integer (popularity, key, time\_signature), boolean (explicit), and text (artist name, track name, album name).  
* **How we use it:** We compute average audio feature profiles per genre (e.g., the average danceability, energy, valence, etc. for "jazz" vs "club" vs "acoustic"). These genre-level audio profiles become our target space. The model learns to map a venue's features to a point in this audio feature space, and we then retrieve the closest genre(s) and sample songs from them.

**Ground Truth Labels**
There is no existing labeled dataset that pairs specific venues with ideal playlists, that pairing is exactly what we're trying to learn. Instead, we construct pseudo-labels as follows: we use the venue's text fields (review\_summary, generative\_summary) and categorical attributes to assign each venue a target audio profile. For example, we define a simple rule-based mapping where keywords like "lively," "loud," and "energetic" in reviews push toward higher energy/danceability targets. We also use price level and category as additional signals (e.g., sports bars get mapped toward rock/pop). These mapped audio vectors serve as the regression targets for supervised training. We understand that this is an imperfect labeling strategy and will evaluate how sensitive results are to these rules and vary based on experimentation and feature engineering.

#### Data Exploration (see attached figures in [A.1])

Our venue dataset contains 4,484 Manhattan businesses across five categories, with restaurants (1,663) and cafes (883) being the most relevant for music selection (Fig 1). Price level skews heavily toward "Moderate" at 65% (Fig 3), and primary\_type provides useful granularity within categories.

The boolean attributes (Fig 5\) are where the real atmosphere signal lives. Common attributes like serves dessert (96%) and good for groups (95%) are too ubiquitous to differentiate anything. The rare ones, live music (11%), watching sports (11%), allows dogs (18%), are the strongest atmosphere differentiators. However, Fig 6 shows these booleans have 50–73% missing rates, and this missingness correlates with venue type (non-food businesses lack these fields), so we treat missing as an informative "unknown" category rather than imputing.

For the songs dataset (114,000 tracks, 114 genres), Figure 10 shows danceability and valence are well-spread across \[0,1\], while acousticness, instrumentalness, and speechiness are heavily right-skewed. The correlation matrix (Figure 14\) reveals key relationships: energy and loudness correlate strongly (0.76), acousticness is inversely related to energy (-0.73), and popularity is independent of all audio features, supporting its use as a sampling weight rather than a model input. Song popularity skews low with a spike near zero (Fig 12), so we will filter out low-popularity tracks to ensure recommendable songs.

Most critically, Figures 15–17 confirm that genres occupy distinct regions of the audio feature space. The energy-vs-danceability scatter (Fig 16\) shows clean spatial separation between genres. This separation is the core assumption our project depends on: if genres weren't distinguishable by audio features, mapping venues to playlists wouldn't be possible.

#### Preprocessing

**Places dataset:**

* Drop 3 fully-empty columns: pure\_service\_area, neighborhood\_summary, attributions.  
* Boolean attributes have 50–73% missing values. Since missingness correlates with venue type (non-food businesses tend to lack these), we treat missing as a third "unknown" value rather than imputing True/False.  
* Rating: 2% missing;  impute with median (4.5).  
* Price level: ordinal encode as integers 1–4. Missing values get a separate indicator column.  
* Neighbourhood and primary\_type: one-hot encode.  
* review\_summary and generative\_summary: apply TF-IDF vectorization, then reduce to \~50 dimensions with truncated SVD to get a compact text representation.  
* Min-max normalize all final numeric features to \[0, 1\].

**Songs dataset:**

* Drop the rows  with the missing artist/track name.  
* Select the 7 core audio features as the target space: danceability, energy, acousticness, valence, instrumentalness, liveness, speechiness.  
* Normalize tempo (0–243 BPM) and loudness (-60 to 0 dB) to 0–1 scale.  
* Compute per-genre mean audio profiles which will help us create the target possible points and overall final space to streamline the outputs to fit into.

### [4.1.2] Methods and Model Training -> - Methods and Model Training: There is no justification for why the machine learning algorithms are well-suited to solve the problem (-2 points)

We will implement two primary regression methods from scratch using NumPy to ensure full algorithmic transparency and learning. Both address the multi-output prediction problem: given venue feature vector **x** ∈ ℝ^d, predict target audio profile **y** ∈ ℝ^7 (danceability, energy, acousticness, valence, instrumentalness, liveness, speechiness).  
**Method 1: Multi-Output Ridge Regression.** Ridge regression minimizes the regularized least squares objective:  
$$\\mathcal{L}(W) \= |\\mathbf{XW} \- \\mathbf{Y}|\_F^2 \+ \\lambda |W|\_F^2$$  
with closed-form solution $W^\* \= (\\mathbf{X}^T \\mathbf{X} \+ \\lambda I)^{-1} \\mathbf{X}^T \\mathbf{Y}$, where **X** is the (n × d) venue feature matrix, **Y** is the (n × 7\) target matrix, and λ is the regularization strength. Ridge is well-suited because our feature space is wide (one-hot neighborhoods \+ TF-IDF text can exceed d \> 100\) with correlated features; L2 regularization shrinks coefficients without eliminating them, providing directly interpretable weights that reveal which features most influence each audio dimension. Predictions follow: $\\hat{\\mathbf{y}} \= \\mathbf{x}^T W^\*$.  
**Method 2: Two-Layer Neural Network.** The architecture is Input (d) → Hidden 1 (128, ReLU) → Hidden 2 (64, ReLU) → Output (7, sigmoid). Forward pass computes:  
$$\\mathbf{z}^{(1)} \= \\mathbf{X}W^{(1)} \+ \\mathbf{b}^{(1)}, \\quad \\mathbf{a}^{(1)} \= \\max(0, \\mathbf{z}^{(1)})$$ $$\\mathbf{z}^{(2)} \= \\mathbf{a}^{(1)}W^{(2)} \+ \\mathbf{b}^{(2)}, \\quad \\mathbf{a}^{(2)} \= \\max(0, \\mathbf{z}^{(2)})$$ $$\\hat{\\mathbf{y}} \= \\sigma(\\mathbf{a}^{(2)}W^{(3)} \+ \\mathbf{b}^{(3)})$$  
Loss is mean squared error over all 7 outputs; backpropagation computes gradients via the chain rule, with weight updates: $W \\leftarrow W \- \\alpha \\cdot \\frac{\\partial \\mathcal{L}}{\\partial W}$. The neural network captures non-linear interactions that Ridge cannot—e.g., a venue with both "live\_music=True" and "price\_level=expensive" may warrant jazz, while "live\_music=True" and "price\_level=inexpensive" warrants rock. Hidden layers learn complex TF-IDF text feature combinations that linear methods would miss.  
**Inputs and Outputs.** Both methods take the same d-dimensional venue feature vector: encoded price level, rating, 18 boolean flags, one-hot neighborhood and primary\_type, and TF-IDF/SVD text representations. Both output a 7-dimensional audio profile vector.  

Exploratory K-Means Analysis. We may explore K-Means clustering to group venues into K archetypes and assign each cluster a shared audio profile, revealing interpretable venue categories (e.g., upscale cocktail bar, casual brunch). K selection uses elbow method and silhouette score. This serves primarily as an interpretive tool revealing venue archetypes rather than as a deployment candidate due to coarser prediction granularity.  

### [4.1.3] Model Evaluation

**Evaluation Metrics.** We will employ three complementary metrics. *Mean Squared Error (MSE):*  
$$\\text{MSE} \= \\frac{1}{n} \\sum\_{i=1}^{n} \\frac{1}{7} \\sum\_{j=1}^{7} (y\_{ij} \- \\hat{y}\_{ij})^2$$  
Since all features normalize to \[0,1\], MSE values are directly comparable across dimensions and models. Preliminary analysis establishes baselines: mean-prediction yields MSE \= 0.022; random genre assignment yields MSE \= 0.044. Models must fall below 0.022 to demonstrate learned signal beyond baseline.  
*Cosine Similarity:*  
$$\\text{CosSim}(\\mathbf{y}, \\hat{\\mathbf{y}}) \= \\frac{\\mathbf{y} \\cdot \\hat{\\mathbf{y}}}{|\\mathbf{y}| \\cdot |\\hat{\\mathbf{y}}|}$$  
Liveness (σ \= 0.045) and speechiness (σ \= 0.030) have low variance across genres, inflating MSE for directionally correct predictions. Cosine similarity evaluates profile shape independent of magnitude, providing a more faithful atmospheric fit indicator.  
*Top-K Genre Retrieval Accuracy:*  
$$\\text{Acc}@K \= \\frac{1}{n} \\sum\_{i=1}^{n} \\mathbf{1}\[\\text{genre}^\*\_i \\in \\text{top-}K(\\hat{\\mathbf{y}}\_i)\]$$  
This measures whether the K nearest genre centroids to the predicted profile include the target genre by Euclidean distance. We will report Acc@3 and Acc@5 after merging near-duplicate genres (singer-songwriter vs. songwriter, distance 0.000; latino vs. reggaeton, distance 0.019) and restricting to 91 commercially viable genres. This metric most directly reflects end-user value by measuring appropriate music recommendations.  
**Training and Validation.** We will restrict training to restaurants and cafes (n \= 2,546), as gyms and retail stores have 100% missing boolean attributes and \<30% text coverage, making pseudo-label construction unreliable. Data splits: 80/20 into train/test stratified by primary\_type, with the training set further split 80/20 into train/validation under identical stratification. All pseudo-labels are constructed before splitting to prevent leakage. Fixed seed: 42\.  
**Planned Experiments.** For Ridge, we will sweep λ ∈ {0.001, 0.01, 0.1, 1, 10, 100}, selecting the configuration with lowest validation MSE. Large λ induces high bias; small λ recovers standard least squares with high variance risk. We will ablate across three feature configurations, boolean flags only, TF-IDF/SVD text only, and full vector to quantify modality contributions.  
For the neural network, we will perform a grid search over learning rate α ∈ {0.001, 0.005, 0.01}, batch size ∈ {32, 64, 128}, and dropout rate ∈ {0.0, 0.2}, training up to 500 epochs with early stopping (patience \= 20\) on validation MSE. Learning rate controls step size; values too large cause oscillation or divergence, while small values slow convergence—we will plot learning curves to diagnose failure modes. Batch size controls gradient update samples; smaller batches introduce noise that escapes local minima but increases loss variance.  
**Regularization and Overfitting.** Ridge, λ, directly governs the bias-variance tradeoff; validation sweep identifies the optimal point. For the neural network with \~47,000 parameters on \~1,629 samples, regularization is critical. Dropout after each hidden layer randomly deactivates neurons during forward passes to prevent co-adaptation. Early stopping halts training when validation MSE fails to improve for 20 consecutive epochs, preventing pseudo-label memorization. We will monitor the training-validation MSE gap. ReLU activations maintain non-zero gradients for positive inputs, addressing vanishing gradients. If both losses plateau high (underfitting), we will apply batch normalization and increase hidden layer width from (128, 64\) to (256, 128).

### [4.1.4] Model Deployment

We will select the deployed model via a three-tier decision procedure on held-out test results, reflecting tradeoffs between Ridge, the Neural Network, and K-Means.  
**Primary Criterion: Top-K Genre Retrieval Accuracy (Acc@5).** This directly corresponds to end-user value by determining whether the system recommends musically appropriate genres. We will require a margin \> 0.03 (approximately 15 venues in our 509-sample test set) to declare a clear winner; smaller differences do not justify deploying a more complex model.  
**Secondary Criterion: Cosine Similarity.** If Acc@5 scores fall within margin, we will select on cosine similarity. This better captures perceptual atmospheric fit than MSE given low discriminative variance in liveness and speechiness. A model producing the correct profile shape—high energy and danceability for a sports bar, high acousticness for an upscale café—generates superior playlists even with marginally higher MSE, reflecting the principle that bias-variance management must be evaluated relative to downstream task, not training loss alone.  
**Practical Tiebreaker: Ridge Regression.** If all metrics remain comparable, we will default to Ridge. Ridge produces predictions via single matrix multiplication without iterative forward passes, satisfying Streamlit's low-latency demands. The weight matrix W\* ∈ ℝ^(d×7) is directly interpretable per output dimension, allowing the application to surface human-readable feature-importance explanations. Ridge degrades gracefully under missing features, routine in live Google Maps API responses where boolean attributes are frequently absent.  
K-Means produces coarser predictions—every venue in a cluster receives identical audio profiles—reducing atmospheric resolution compared to continuous Ridge and Neural Network predictions. Thus it serves primarily as an interpretive tool revealing venue archetypes rather than as a deployment candidate.  
**Deployment Pipeline.** The selected model will be serialized into the Streamlit application. At runtime, a user-submitted Google Maps link triggers preprocessing to construct the venue feature vector, which feeds into the model. The model outputs the predicted audio profile; we will retrieve the K nearest genre centroids by Euclidean distance in the 7-dimensional audio space and sample songs weighted by Spotify popularity. The application will display the predicted profile as a radar chart showing all seven acoustic dimensions, enabling venue operators to understand which features drove the recommendation without music or data science expertise.

## [4.2] Front-End Application Using Streamlit

The **target population** is business owners and managers of venues like restaurants, hotels, retail stores and fitness centers who require an automated music curation solution without prior knowledge of machine learning, currently served by services like Soundtrack Your Brand or Mood Media.

The UI layout is depicted in Figures 19, 20 and 21\. The sidebar contains navigation buttons for Home, About Us, Recently Generated, Most Played, Your Favorites, Settings and Logout. The top bar includes Contact Us, How It Works and Sign Up. The primary input is a Google Maps text field accompanied by a Generate Playlist button that triggers the back-end pipeline and returns a curated playlist. After clicking Generate Playlist the application renders two pages: the Statistics page displaying venue analysis including busyness patterns, category and extracted attributes, and the Playlist page displaying song titles, artists, energy and valence after the user clicks GO TO PLAYLIST.

All three pages share a consistent sidebar organized into four sections: Menu, Library, Playlist and Favorite and General. Figure 19 shows the Home page with a hero banner, Google Maps input field and Generate Playlist button. Figure 20 shows the Statistics page with three metric cards, Vibe Tags, Sentiment Breakdown, Busyness by Hour sliders and Acoustic Targets panels and a GO TO PLAYLIST button. Figure 21 shows the Playlist page with a header and 20 ranked tracks. Figure 22 is a flowchart of the full system architecture. When the user clicks GENERATE PLAYLIST, \\texttt{st.button()} triggers a full re-execution of the Streamlit script with settings persisted via \\texttt{st.session\_state}. The Google Maps Places API retrieves venue data while the Spotify Tracks Dataset is loaded as the candidate song pool. Both feed into preprocessing to produce a feature vector passed to Ridge Regression, a Neural Network and K-Means Clustering built from scratch using NumPy. The Statistics page renders results using \\texttt{st.bar\_chart()}, \\texttt{st.slider()} and \\texttt{st.metric()}. GO TO PLAYLIST updates \\texttt{st.session\_state} and navigates to the Playlist page rendered using \\texttt{st.dataframe()}. AtmoSound follows a linear three page flow with one decision per page, busyness sliders for real time override and a consistent dark theme with pink accents suited to the music and hospitality industry.

**Front-End connection to Back-End (Website Layout and Functionality)):** All three pages share a consistent sidebar containing the AtmoSound logo and navigation links organized into four sections: Menu (Home, About Us), Library (Recently Generated, Most Played), Playlist and Favorite (Your Favorites) and General (Settings, Logout). Figure 19 shows the Home page featuring a full width hero banner with a Google Maps URL input field and a Generate Playlist button, with two feature cards below. Figure 20 shows the Statistics page with three metric cards showing average rating, review count and peak busyness, and four panels showing Vibe Tags, Sentiment Breakdown, Busyness by Hour sliders and Acoustic Targets, with a GO TO PLAYLIST button at the bottom. Figure 21 shows the Playlist page with a header displaying playlist metadata and a full width track listing of 20 rows showing album art, song title, artist, genre, album, date and duration. Figure 22 is a flowchart showing the full system architecture connecting the front end and back end through a Streamlit pipeline.

The pipeline starts when the user pastes a Google Maps URL and clicks GENERATE PLAYLIST, implemented using \\texttt{st.button()} which triggers a full re-execution of the Streamlit script with settings persisted via \\texttt{st.session\_state}. The Google Maps Places API retrieves venue data including rating, price level, primary type, opening hours and place category, while the Spotify Tracks Dataset is loaded as the candidate song pool. Both feed into preprocessing to produce a feature vector passed to Ridge Regression, a Neural Network and K-Means Clustering built from scratch using NumPy, whose combined outputs generate a final ranked track list. The Statistics page renders venue attributes, vibe tags, sentiment breakdown, busyness sliders and acoustic targets using \\texttt{st.bar\_chart()}, \\texttt{st.slider()} and \\texttt{st.metric()}. The user clicks GO TO PLAYLIST which updates \\texttt{st.session\_state} and navigates to the Playlist page rendered using \\texttt{st.dataframe()}. AtmoSound follows a linear three page flow ensuring the user faces one decision at a time, with the busyness sliders allowing real time override of historical data and the consistent dark theme with pink accent colors creating a professional interface suited to the music and hospitality industry.

| Figure | Question          | Description                                    |
| :----- | :---------------- | :--------------------------------------------- |
| Fig 1  | 4.2 (Question 2\) | Home Page Layout                               |
| Fig 2  | 4.2 (Question 5\) | Statistics Page Layout                         |
| Fig 3  | 4.2 (Question 5\) | Playlist Generated Layout                      |
| Fig 4  | 4.2 (Question 5\) | Flowchart for Front-End to Back-End connection |


[Figma](https://www.figma.com/design/9u4ravj5uUmg4kcjaj0IFI/AtmoSound-Final-Proposal?node-id=1-4&t=LUSxelZiGufd4wv5-1%20)

# [5] Risk & Mitigation

Three anticipated challenges are Google Maps API rate limits and incomplete venue profiles reducing analysis quality, the model relying on historical busyness data that may not reflect the actual current atmosphere such as generating high energy music on a quiet Friday evening and the Spotify Tracks Dataset being three years old and potentially lacking recent songs. To mitigate these, the team will cache Google Maps data early in development, provide busyness sliders on the Statistics page for real time operator override and filter the Spotify dataset to prioritize tracks from the past five years.
# [6] **Appendix 

## [A.1 - 4.1.1] Diagrams (4.1.1 Data Collection, Exploration & Processing (6 points) in Grading Rubric)

|        \[1\] Venue Category Distribution        |        \[2\] Venue Rating Distribution        |
| :---------------------------------------------: | :-------------------------------------------: |
|                   ![][image1]                   |                **![][image2]**                |
|       **\[3\] Price Level Distribution**        |    **\[4\] Venue Count by Neighbourhood**     |
|                 **![][image3]**                 |                **![][image4]**                |
|     **\[5\] Boolean Attribute Prevalence**      | **\[6\] Boolean Attribute Missingness Rates** |
|                 **![][image5]**                 |                **![][image6]**                |
|   **\[7\] Geographic Distribution of Venues**   |      **\[8\] Venue Rating by Category**       |
|                 **![][image7]**                 |                **![][image8]**                |
|        **\[9\] Primary Type Frequency**         |    **\[10\] Audio Feature Distributions**     |
|                 **![][image9]**                 |               **![][image10]**                |
| **\[11\] Audio Feature Distributions by Genre** |    **\[12\] Song Popularity Distribution**    |
|                **![][image11]**                 |               **![][image12]**                |
|          **\[13\] Tempo Distribution**          |  **\[14\] Audio Feature Correlation Matrix**  |
|                **![][image13]**                 |               **![][image14]**                |
|   **\[15\] Genre-Level Mean Audio Profiles**    |  **\[16\] Energy vs. Danceability by Genre**  |
|                **![][image15]**                 |               **![][image16]**                |
|     **\[17\] Genre Audio Feature Heatmap**      |    **\[18\] Venue Rating by Price Level**     |
|                **![][image17]**                 |               **![][image18]**                |


## [A.2 - 4.2] FIGMA (4.2 Front-End Application Using Streamlit in Grading Rubric)

[Figma](https://www.figma.com/design/9u4ravj5uUmg4kcjaj0IFI/AtmoSound-Final-Proposal?node-id=1-4&t=LUSxelZiGufd4wv5-1%20)

**Figure 19 \- UI**

**Figure 20 \- UI**

**Figure 21 \- UI**

**Figure 22 \- 4.2 (Question 5\)**
