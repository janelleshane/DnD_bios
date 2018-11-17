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

### 0.5 temperature example ###

note: there are no breaks between characters at this temperature. Generated text appears to be the middle of a single neverending bio.

```
them. he was a great deal of fate and a man who used to create his head. the city of the elder brain in the desert and three years of adventure and her father had recently really a different world. when she was 16, and he was a child before the only thing she was a strong sense of his home and the god of the arcane arts, and the inquisitor was not a chance to the next few years of the world, a small group of adventurers and the underdark that they were not alive for the great group of adventurers and many of the family heirloom and interesting. he was with a demon of the monastery that he had to return to the monastery to remove the songs of the village and passed the sword. he had a great deal of intelligence and her family were the youngest of 19 and the other time he was a respectable in the middle of the great city. he was only a young age of the underdark as a child. it was no more than he was a kind and a number of the family who had been a licensed explosion. a few days later, she knew the truth by a human woman who came to a powerful magical contract and control the ship and the lady of his parents. the halfling was a child of 12, and as a result of the lives of the human family of his shoulder, the young elf of the same time the worlds widows wears the first time in the wilds of the gods and her father was in a particularly connected of a demon of the clan. he awoke up my body. the clan was in the world to find himself. he had to return to the surface world. he was the first time they were still aware of the village to keep him a new identity in the big city of his home and were forced to be a wizard, and has an honest in the world and the result of a confidence and a fellow gnome who had been sent out in the city of droum. his father was a new child, and was a bit of a new reminder of the eldritch of the world with a noble family of a human bard, he set out to make people and his parents were making a pack of losing the less of the city of orcs for the ruins of the way of the greatest other slaves and her parents had a spark of power, and he had a great deal of adventurers to be able to be in the world. he was not as a reminder of the lore to a new family, which was unsuited to the village to seek out the ship and decided to be a paladin of the side of a secret and remaining from the other town, and became a paladin of a major city of river to the conversation of the undead of the river. he was 15, he began to make sure to do it. the city of lightning bolt was discovered by the dark moon of the monastery on the ground. she was a half-elf who lived from the fight for the guards to help him a living as a reminder of the town, she had a god to find his own home that often got a back of faerun of his life that was not an understanding of the fire to the interest in the world to be a world to remove the world of the elves, he was a bit of a small group of trades. she was a constant violent predator rabbit awkwardprobe bleep echo clot blob violent predator rabbit awkwardprobe bleep echo clot blob violent predator rabbit awkwardprobe bleep echo clot blob violent predator rabbit awkwardprobe bleep echo clot blob violent predator rabbit awkwardprobe bleep echo clot blob violent predator rabbit awkwardprobe bleep echo clot blob violent predator rabbit awkwardprobe bleep echo clot blob violent predator rabbit awkwardprobe bleep echo clot blob violent predator rabbit awkwardprobe bleep echo clot blob violent predator rabbit awkwardprobe bleep echo clot blob violent predator rabbit awkwardprobe bleep echo clot blob violent predator rabbit awkwardprobe bleep echo clot blob violent predator rabbit awkwardprobe bleep echo clot blob violent predator rabbit awkwardprobe bleep echo clot blob violent predator rabbit awkwardprobe bleep echo clot blob violent predator rabbit awkwardprobe bleep echo clot blob violent predator rabbit awkwardprobe bleep echo clot blob violent predator rabbit awkwardprobe bleep echo clot blob violent predator rabbit awkwardprobe bleep echo clot blob violent predator rabbit 
```

### 0.8 temperature example ###

note: line breaks added after `""` symbols to help separate the individual bios. Bios start with character name, race, and class (no spaces between these)

```
"meleskithhalf-orcbarbarian"am allow over that emizah was income due to murder the illithid who have went fun of mortals. i told her up with her by the musical noble), not until he looked crazy for the most lands for her as i was drugs of problems and anwin rider; who had been in 15 year of bandits in these with them. his words are a problem to take up his studies and ran back to the gang of the second of the world and took him that she would happen or too care of his studies and sister, the daughter of the clan, as well as the sea as the key of the tribe. she was there that the larger farm is gone on him (or reasons. he is sure that the children were discovered by troutory, and when he was flung in the wilderness with largely again as she got to be a licensed orphaned by tacking a warlock. she was a young age -- be inflicteding the city his ability to assist the stuff of the lizard in the goliath tribes of hidden temple of the family as the tribe called her birth and help in each other as either to ask to become a purple blank, and the best brothers were even more up and youngest of determination. the goddess of st and crime behind the creature of his homeland and the heavily and fairly kind of a song that can and his father was attacked by two complicated child.tyr began to the point and scrounged a home /nonisms of situator, they also saw me destroyed by costs, but his father had more an unknown coptious threats, along with herness and now he learned how to go from his face. he awoke to meet my freedom. most of the cleric of the city she received the respect who had been priests for adventure.

"chef tidhalf elffighter"leusandra was who a large time in vain to shoot a human. after being watch, he was an elf, a bag battles. the shops of the head of the thought of the grey rat that most often sleep. he has a child that was healed from and instilled the country, and he threw her and often tactics with a jagged responsible society, further into the health. he considered her parents to bad basing other power. he was discovered, he has recently be deserve, a pouch later in the end of his life, so she felt not to be a special magic and relief his mother. her father leveraged to go with them from the peaceful face, and letten and with whom they were quite life to bleed larger child. he was born on a game of hell, and realized that he was done off with me for a particularly fragment. with then, his god lords - this personalities when she was now as different money and ordered a brief two honers, in charge of the swamp of his wife to ""i medalent and a busy time to be very good to his sad mac. i learned a child in a man he used to live with him and he wanted to use the quiet manners to see his duty in his clan. it was a rough time to return to perform praise.
```
