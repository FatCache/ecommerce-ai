### Problem Statement

use dataset from huggingface amazon reviews 2023


Human input and AI Bot returns recommendation
input: `What should I wear to the beach`?

machine response needs to be human like with recommendation sourced from hugging face amazon review dataset

keep problem within scope: focus fashion only

additional challenge: work around situation where dataset is too big to load, how to circumnavigate this?

#### Run Server

`fastapi dev .\server.py`


#### Two collections
a. meta, only consider main_category, title, average rating, feature, description, price
b.

#### Search Strategies
Given a query, fundamental step is price. 
a. If price is provided - find anything at less than or equal to it [uses meta data]

Breaking down into two collection for now means we will apply federated search. 

#### Retrival
a. returns 'reviews' closet to what the user said? like, "looking for green chair". Returns review like "goes well with my green room"
    <this means search needs to drill down to product and finds reviews most revelant to the search, what if only single review ??>

todo: 
- big one - on basis of what field should i build chroma embedding on

#### Technical
- embedding model to use: text-embedding-3-small/large ?