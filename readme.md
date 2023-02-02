# Furniture Products Detection Assignment

This repository presents a possible solution for the given assignment: *extracting product names from a list of given websites*. The approach builds
on a pretrained **DistilBert** (uncased) as a **backbone** with 2 inserted FC layers and a `[CLS]`-based **text classification** training objective. To me, this made more sense than a pure **NER**-oriented approach and I will try to explain why.


A demo is available as a Google Colab Notebook!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14VOdhVGyv9AxmcBoS2EZ43f0zygESDUv?usp=sharing)


## Insight and Observations

The following observations are used to justify some of the design decisions in my solution:

### Websites Related

1. The presented data is **semi-structured** since it must match the HTML structure
2. The websites match the **e-commerce** website **template** -> they are likely to meet some basic SEO / good indexing & searching requirements
while also matching the HTML standards
    1. They have a **separate page** for **each product**
    2. Important information (such as **product name**) is presented **statically** (i.e., not JavaScript loaded) to help search engines find that information easier
    3. In-page information is presented hierarchically through **HTML headings**; i.e., the semantic meaning (w.r.t. a page's content) decreases when going from `<h1>` (possibly product name?) to `<h5>` (generic "contact us" options).


### Model Related

1. The **content** is mostly written in **English**
2. The **names of products** must be learned and also:
    1. Account for possible **variations** (e.g., suffixes/prefixes - *stool vs barstool*, randomly inserted symbols / weird formatting, etc.) -> some sort of tokenization is needed
    2. Account for **semantically similar names** (e.g., chair ~ stool) -> a large number of pretrained embeddings is needed
3. **Casing** seems to be irrelevant - *king bed* and *King Bed* should be both matched
    1. From a dataset perspective, it becomes relevant since a larger dataset and different augmentation strategies are needed to account for variations introduced by casing


### Conclusions

By leveraging **HTML**-based information from each website's sourcecode it is easy to notice that this is not necessarily a **NER** problem. There's a fairly good chance
that the data is **already segmented** (or delimited) through different **tags**. 

Presumably, there's a small chance of encountering a page such as:
```html
<body>TheBestShop is selling Westlake Queen Storage Bed at $499</body>
```

But the following is more likely:
```html
<body>
    <h1>TheBestShop</h1> 
    <h2>Westlake Queen Storage Bed</h2>
    <span>$499</span>
</body>
```

This **decreases the problem's complexity** (the model doesn't have to also learn to isolate entities from a "flat" & noisy text input) and in turn *should* permit **better performance** by delegating the segmentation task to the **web crawler**. At the same time, it is **more favourable** to **annotate datasets** for a simple classifier. 

Besides this, there's the **reduced sequence length** which avoids a signifiant limitation of the transformer's  attention model; i.e., the NER input is expected to be larger than a possible product name - this is forcing the transformer's multi-head attention to use more memory or even require input truncation.

By also taking into account the fact that most websites present the needed information statically -> a fast HTML-only **web crawler** can be used.

A **DistilBert** model make sense in this scenario because:
1. a **checkpoint** is available for the **uncased** version
2. is **fast** (training & inference wise) -> could provide **better throughput**


### FAQ

> Why not a more recent model such as DistilRoBERTa?

I couldn't find a checkpoint for the uncased version of DistilRoBERTA; running the cased version would probably require a larger dataset.

> Why a distilled model?

The task at hand is relatively simple: doing something like `product_name.contains("bed")` will yield some decent results for beds. However,
the number of websites is large. At the same time, DistilBert is said to be "40% smaller than the original BERT-base model, 60% faster than it, and retaining 97% of its functionality", which is adequate for a higher-throughput approach.

> What if a products list / category is given and it's not using HTML headings?

In the current implementation, the crawler supports a BFS-like indexing; if the assumption that each product has its own page holds
then the crawler will find a link to the product's page which will be properly formatted (i.e., have the necessary HTML headings).


## Web Crawler

The web crawler (*crawler.py*) relies on the **requests** and **BeautifulSoup4** modules for getting the source code and parsing it.
It implements the following functionalities:
1. basic web **page level crawling** and extraction of **tags of interest** (e.g., `<h1>`, `<h2>`, ...)
2. **BFS-based website crawling** by analyzing links within the sourcecode; **depth** and **maximum number of pages** are configurable parameters
3. **relative** -> **absolute** URL conversion by relying on **initial hostname**
4.  modified / realistic **user agent** to avoid some possible bot filters

### FAQ

> Why not Selenium?

Selenium (w/ headless Chrome Driver) was my first choice since it also covered JavaScript-based actions but the crawling process was too slow (despite trying to improve it). 
This made me reconsider the necessity of taking JavaScript into account for such websites.

## Dataset

Manual annotation of data by starting from scratch is a time consuming, almost infeasible, process. Even with a pretrained & frozen backbone,
many samples must be provided to successfully capture the semantics of a product name.

Therefore, I kept the provided list of websites strictly as a **testing set** and looked for **existing datasets** which provided names of products
from the same field (furniture, household items). At the same time, I crawled a part of English **wikipedia** to gather word sequences for **negative sampling**.

### Employed Datasets

The following datasets were downloaded, preprocessed, adapted and used for **training** and **validation**:
1. [IKEA SA Furniture](https://www.kaggle.com/datasets/ahmedkallam/ikea-sa-furniture-web-scraping)
2. [IKEA US Products Dataset](https://www.kaggle.com/datasets/crawlfeeds/ikea-us-products-dataset)
3. [SOUQ Office Furniture Dataset](https://www.kaggle.com/datasets/marwahmm/souqcom-dataset)
4. [Furniture Images Dataset](https://www.kaggle.com/datasets/lasaljaywardena/furniture-images-dataset)
5. [Flipkart Furniture Dataset](https://www.kaggle.com/datasets/neerajjain6197/flipkart-furniture-dataset)
6. Random English Wikipedia Articles (crawled by me)

### Details

There should be about **281,072** annotated sequences, divided as:
- **training samples:** 234,463
- **validation samples:** 46,609

The datasets are stored in a **JSON** format. For example, below are representations for a **positive sample** followed by a **negative sample**:
```json
{"Office chair with armrests": true, "Box - black 9 ¾x13 ¾x9 ¾": true}
{"Russian at Amherst": false}
```

Each **negative sample** is generated by randomly sampling sequences of different lengths from the Wikipedia Corpus (dataset #6 from above). Additionally,
random (non-sequential) sampling is performed at loading time to further expand this dataset.

**Dataset imbalance** (or **precision-recall** tradeoff) is tweaked by modifying the loss penalty for positive samples at training time.

### Augmentation Methods

Three runtime augmentation methods are implemented, to further provide more dataset diversity:
1. **Random Elements Deletion:** from a given sequence deletes between 0 and 3 elements (words) to force the model to establish context without relying purely
on specific words
2. **Random Noisy Element:** replaces a sample with a random sequence of ASCII printable characters and labels it as a negative sample; this is added to hopefully account for some unexpected words / artifacts
3. **Random Trimming:** highly specific augmentation method which sometimes splits a product's name by the `-` (minus/dash) character to account for some formatting biases

## Transformer Model

The model relies on Huggingface's pretrained `DistilBertModel` & `DistilBertTokenizer` (`distilbert-base-uncased`) with **frozen weights**.
The last hidden state of DistilBert is shaped as `[batch_size, seq_len, hidden_size]`, but only the first element is of interest. 

In PyTorch, two fully connected layers are attached to the `[CLS]` token in order to map its representation from **768** floats to **1** (an is_product() logit).
The other tokens in the sequence are discarded.

### Training Parameters

- **Batch size:** 128 (dynamic padding)
- **Initial LR:** 3e-4 (linear decrease to 3e-6) w/ Binary Cross Entropy
- **Epochs:** 40
- **Optimizer:** AdamW w/ default weight decay regularization coefficient (1e-2)
- early stopping

## Results

Numeric values are available in this Google Sheet: [Furniture Transformer Stats](https://docs.google.com/spreadsheets/d/1iTmqe3nxbfkBlEJHDKh9jAyaIOdPzeN2KpzP8Z7HJlw/edit?usp=sharing)

### Precision - Recall

**Precision** and **Recall** are tracked (per epoch) during the **validation** stages and are observed to have similar values.

![2D Evolution of Precision-Recall](imgs/precision-recall-2d.png?raw=true "2D Evolution of Precision-Recall")

![Precision, Recall F1 Evolution](imgs/precision-recall-f1.png?raw=true "Evolution of Precision, Recall and F1")


### BCELoss Curves

The plot below tracks the possible increase in variance (overfitting) of the model for the training & validation sets
![BCELoss Curves](imgs/bce_loss.png?raw=true "Training vs Validation Loss")


### Qualitative Results on Testing Set

Below are some qualitative results for the provided websites with the mention that **the module identifies products from their corresponding pages**.
If the crawler never reaches the main product's page, the product's name won't be registered; therefore, the crawler must run in "BFS-crawling mode".

**Note:** the results also include examples which are marked as non-products to show that the model discriminates between a product's name and non-interesting strings and it's not just validating everything from a HTML heading.

See the: `outputs/` dir for more outputs.

```JSON
"https://www.factorybuys.com.au/products/euro-top-mattress-king": [
    {
        "query": "Factory Buys 32cm Euro Top Mattress - King",
        "is_product": true,
        "confidence": 0.9997435212135315
    }
],
"https://dunlin.com.au/products/beadlight-cirrus": [
    {
        "query": "Beadlight Cirrus LED Reading Light",
        "is_product": true,
        "confidence": 0.9954960346221924
    }
],
"https://themodern.net.au/products/hamar-plant-stand-ash": [
    {
        "query": "Hamar Plant Stand - Ash",
        "is_product": true,
        "confidence": 0.9862573742866516
    },
    {
        "query": "Hamar Plant Stand - Ash",
        "is_product": true,
        "confidence": 0.9862573742866516
    }
],
"https://furniturefetish.com.au/products/oslo-office-chair-white": [
    {
        "query": "Sorry, this shop is currently unavailable.",
        "is_product": false,
        "confidence": 0.9740400388836861
    },
    {
        "query": "Did you mean ?",
        "is_product": false,
        "confidence": 0.9999966273669543
    },
    {
        "query": "Only one step left!To finish setting up your new web address, go to\n              your domain settings,\n              click \"Connect existing domain\", and enter:",
        "is_product": false,
        "confidence": 0.9999999833593751
    }
],
"https://hemisphereliving.com.au/products/": [
    {
        "query": "No Results Found",
        "is_product": false,
        "confidence": 0.999991578429217
    }
],
"https://home-buy.com.au/products/bridger-pendant-larger-lamp-metal-brass": [
    {
        "query": "Oops - We couldn't find that one",
        "is_product": false,
        "confidence": 0.9990170887904242
    }
],
"https://interiorsonline.com.au/products/interiors-online-gift-card": [
    {
        "query": "Interiors Online Gift Card",
        "is_product": true,
        "confidence": 0.9939302206039429
    },
    {
        "query": "Australia's Exclusive Online Furniture Store",
        "is_product": false,
        "confidence": 0.6729002296924591
    }
],
"https://beckurbanfurniture.com.au/products/page/2/": [
    {
        "query": "Products",
        "is_product": false,
        "confidence": 0.9959009154699743
    },
    {
        "query": "Explore Our Range",
        "is_product": false,
        "confidence": 0.9999735145356681
    }
],
```

### Outputs

The `outputs/` directory includes 3 files with generated outputs:
1. **mixed_depth0_50.json** - for each given url, presents found products and rejected texts (i.e., texts found in HTML headings but which
do not represent products); can be used to evaluate how the model performs
2. **products_depth1_50.json** - for each given url, it detects the products from the page and attempts to crawl another
49 pages from the same website; can probably serve as a decent dataset for similar tasks
3. **products_depth0_50.json** - this looks for products only in the given urls, without extending the crawling process



## How to run

Simply install the required dependencies via **pip3**:

```pip3 install -r requirements.txt```

Download the last checkpoint [4.168538576697756.dat](https://drive.google.com/uc?id=1g7I0Q1L0DxZY-iGETFsa-9daA_Ds6aCC&export=download) and place it in the `checkpoints/` directory.


Then simply run:

```python3 run.py```

Check out the Colab for more details.






























