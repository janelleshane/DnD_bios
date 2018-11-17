# DnD_bios
Crowdsourced dataset of D&amp;D character biographies and results of syllable-level RNN training on this dataset.

## Crowdsourced dataset

Crowdsourced between 9/17/2017 and 10/24/2018 from readers of aiweirdness.com

2,430 total responses (with spam removed)

Character-only listings not removed.


Original prompt text:
```
I train neural networks to write humor by giving them datasets that they have to mimic. You can check out my work at lewisandquark.tumblr.com 

Recently I trained a neural network to write Dungeons and Dragons spells, and got fun results such as "Hail to the Dave", "Charm of the Cods", and "Curse Clam". http://lewisandquark.tumblr.com/post/165373096197/a-neural-network-learns-to-create-better-dd

Recently I asked for readers to submit the names of their D&D characters so we could find out what happens when the neural network tries to generate character names. The response was great, a few thousand submitted within the first day - I'll definitely be using this dataset! (If you want to contribute to the character names project, go to this form: https://docs.google.com/forms/d/e/1FAIpQLSerLszjv3H5ER4-mrV-0RO9beOAvfC2eqtqGU_izcYpNe5boA/viewform )

It turns out there are researchers working on training neural networks that can tell coherent stories: https://motherboard.vice.com/en_us/article/ypwykx/this-ai-creates-interactive-fiction-by-reading-other-peoples-stories
Someday these might be used to help write really, really open-ended video games. The stories in the article above, though, were about going to the movies or robbing a bank. Let's insert some dragons!

We're going to build a dataset of D&D character backstories. If we get enough, researchers will be able to use them for training a new generation of storytelling AI.

So, use the form below to tell the neural networks about your character's backstory. Or about your friend's character's backstory. You can submit as many of these as you want. No length limit - in fact, details are encouraged. For science!
```

## 150k syllables of RNN output trained on the above dataset

The RNN: torch-rnn by @learningtitans

https://github.com/learningtitans/torch-rnn/blob/valle-syllables/doc/flags.md#preprocessing

Used in syllable-level mode, American English dictionary

Training checkpoint: 11000

Training parameters used:
`-model_type lstm`
`-num_layers 3`
`-rnn_size 512`
`-seq_length 30`

Sampling temperatures:
`0.5` and `0.8`
