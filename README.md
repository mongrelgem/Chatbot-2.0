# Chatbot-2.0
_A Recurrent Sequence to Sequence, multi-domain generative conversational model chatbot implemented in pytorch_

![alt text](https://blog.snapengage.com/wp-content/blogs.dir/3/files/2018/07/chatbot-blog-banner-72618.png)

Overview: 

* Handle loading and preprocessing of [Movie-Dialogs Corpus dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) 
* Implement a sequence-to-sequence model with [Luong attention mechanism(s)](https://arxiv.org/abs/1508.04025)
* Jointly train encoder and decoder models using mini-batches
* Implement greedy-search decoding module
* Interact with trained chatbot

The Brains of the chatbot is a sequence-to-sequence(seq2seq) model. This takes a variable-length sequence as an input and returns a variable-length sequence as an output using a fixed size model. By using two separate recurrent neural nets together, we can accomplish this task. One RNN acts as an encoder, which encodes a variable length input sequence to a fixed-length context vector. In theory, this context vector (the final hidden layer of the RNN) will contain semantic information about the query sentence that is input to the bot. The second RNN is a decoder, which takes an input word and the context vector, and returns a guess for the next word in the sequence and a hidden state to use in the next iteration.

Example Output:

hello?<br/>
`Bot: hello.`<br/>
where am I?<br/>
`Bot: you re in a hospital.`<br/>
who are you?<br/>
`Bot: i m a lawyer.`<br/>
how are you doing?<br/>
`Bot: i m fine.`<br/>
are you my friend?<br/>
`Bot: no.`<br/>
you're under arrest<br/>
`Bot: i m trying to help you !`<br/>
i'm just kidding<br/>
`Bot: i m sorry.`<br/>
where are you from?<br/>
`Bot: san francisco.`<br/>
it's time for me to leave<br/>
`Bot: i know.`<br/>
goodbye<br/>
`Bot: goodbye.`<br/>


For a detailed explanation, go through the repo in the following order: 
1. Preparing the Data
2. Defining the Models
3. Training & Evaluation


This project implements ideas from the following papers:
* [Sutskever et al.](https://arxiv.org/abs/1409.3215)
* [Bahdanau et al.](https://arxiv.org/abs/1409.0473)
* [Luong et al.](https://arxiv.org/abs/1508.04025)

Acknowledgements:
* [Yuan-Kuei Wu’s pytorch-chatbot implementation](https://github.com/ywk991112/pytorch-chatbot)
* [Sean Robertson’s practical-pytorch seq2seq-translation example]( https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation)
* [FloydHub’s Cornell Movie Corpus preprocessing code]( https://github.com/floydhub/textutil-preprocess-cornell-movie-corpus)
* [Matthew Inkawhich's basic tutorial](https://github.com/pytorch/tutorials/blob/master/beginner_source/chatbot_tutorial.py)
