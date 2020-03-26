## Hybrid Recommender

This repo contains notebooks to train a hybrid recommender that incorporates user and item features, including image and text features:
1) Use VGG16 to create image embeddings for item images: [Generate image embeddings](https://github.com/us-ocp-ai/HybridRecommender/blob/master/notebooks/1_generate_image_embeddings.ipynb)
2) Train an autoencoder and apply it to VGG16 embeddings for dimensionality reduction: [Autoencoder](https://github.com/us-ocp-ai/HybridRecommender/blob/master/notebooks/2_autoencoder.ipynb)
3) Train recommendation model: [Hybrid Recommender](https://github.com/us-ocp-ai/HybridRecommender/blob/master/notebooks/3_hybrid_recommender.ipynb)
