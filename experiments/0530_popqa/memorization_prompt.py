answer_prefix_extraction_prompt = """\
You task is to extract a prefix from an article that satisfies some criterion which we will describe later. Specifically, you'll be given a wikipedia ARTICLE enclosed in triple backticks ```___article___```. This article is a reference for a factoid question that asks about the identity of an OBJECT with RELATION to SUBJECT. You will be provided the SUBJECT and any possible aliases, as well as the OBJECT and any possible aliases. You will also be provided the RELATION. You will be provided the QUESTION. And possible ANSWERS (which are essentially aliases to OBJECT).

Respond with an exact prefix of the ARTICLE (meaning even if there's a typo, you copy it exactly, and copy any whitespace), such that including even the next word would start revealing the ANSWER. Respond with only the prefix, and nothing else. Don't apologize or chit-chat, I'm just trying to extract the prefix.

Here's a first example:
SUBJECT: ['Back Home']
RELATION: producer
OBJECT: ['Chuck Berry', 'Charles Edward Anderson Berry', 'Charles Edward Anderson Chuck Berry']
QUESTION: Who was the producer of Back Home?
ANSWERS: ["Chuck Berry", "Charles Edward Anderson Berry", "Charles Edward Anderson Chuck Berry"]
ARTICLE
```Back Home (Chuck Berry album)

Back Home is the fourteenth studio album by Chuck Berry, released in 1970 by Chess Records. The album title refers to his return to Chess after several years with Mercury Records.
Track listing.
All songs written by Chuck Berry.
Cover versions.
"Tulane" was covered by the Steve Gibbons Band in 1977, reaching number 12 on the UK Singles Chart and spending eight weeks in the Top 40. It was also covered by Joan Jett and the Blackhearts on their 1988 album "Up Your Alley" and by Chris Smither on his 1991 album "Another Way to Find You".
"I'm a Rocker" was covered by the British rock group Slade for their 1979 album "Return to Base" and it was the 'inspiration' for AC/DC's "Rocker" on 1975's "T.N.T."
"Christmas" was covered by Clarence Spady; it was released on 11/5/21 as a digital single.
```

For this first example, your response should be:
"Back Home (Chuck Berry album)

Back Home is the fourteenth studio album by"
because the next word starts revealing the ANSWER.

Here's a second example:
SUBJECT: ['Richard Rawlinson']
RELATION: place of birth
OBJECT: ['London', 'London, UK', 'London, United Kingdom', 'London, England', 'Modern Babylon']
QUESTION: In what city was Richard Rawlinson born?
ANSWERS: ["London", "London, UK", "London, United Kingdom", "London, England", "Modern Babylon"]
ARTICLE: 
```Richard Rawlinson

Richard Rawlinson FRS (3 January 1690 – 6 April 1755) was an English clergyman and antiquarian collector of books and manuscripts, which he bequeathed to the Bodleian Library, Oxford.
Life.
Richard Rawlinson was a younger son of Sir Thomas Rawlinson (1647–1708), Lord Mayor of the City of London in 1705–6, and a brother of Thomas Rawlinson (1681–1725), the bibliophile who ruined himself in the South Sea Company, at whose sale in 1734 Richard bought many of the Orientalia. He was educated at St Paul's School, at Eton College, and at St John's College, Oxford. In 1714, he was elected a Fellow of the Royal Society, where he was inducted by Newton. Rawlinson was a Jacobite and maintained a strong support for the exiled Stuart Royal family throughout his life. In 1716 was ordained as a Deacon and then priest in the nonjuring Church of England (see Nonjuring schism), the ceremony being performed by the non-juring Usager bishop, Jeremy Collier. Rawlinson was, in 1728, consecrated as a Bishop in the nonjuring church by Bishops Gandy, Blackbourne and Doughty. On Blackbourne's death in 1741 he became the senior nonjuring Bishop in London, and still maintained a congregation into the mid 1740s. He was particularly concerned with collecting the history and archives of the nonjurors, but later squabbled with his fellow bishops in continuing the succession with the consecration of Robert Gordon. 
Rawlinson travelled in England and on the continent of Europe, where he passed several years, making very diverse collections of books, manuscripts, pictures and curiosities of manuscripts, coins and curiosities, his books alone forming three libraries, English, foreign and Classical. 
Rawlinson was a friend of the antiquarian Thomas Hearne and, among his voluminous writings, published a "Life" of the antiquary Anthony Wood.
Towards the end of his life, Rawlinson quarrelled with both the Royal Society and the Society of Antiquaries. Cutting the Society of Antiquaries from his bequests, he began transferring his collections to the Bodleian. Among his collection was a copperplate known as the Bodleian Plate depicting structures in Williamsburg, Virginia. A series of almanacs in 175 volumes, ranging in date from 1607 to 1747 arrived in 1752–55. At his death, Rawlinson left to the Library 5,205 manuscripts bound in volumes that include many rare broadsides and other printed ephemera, his curiosities, and some other property that endowed a professorship of Anglo-Saxon at Oxford University. The Rawlinsonian Professor of Anglo-Saxon was first appointed in 1795. He was also a benefactor to St John's College, Oxford.
He died at Islington, London.
Richard Rawlinson is buried at St John's College, Oxford, allegedly holding the skull of Christopher Layer, an executed Jacobite.
. Rawlinson Road in North Oxford is named after him.
```

For this second example, your response should be:
"Richard Rawlinson

Richard Rawlinson FRS (3 January 1690 – 6 April 1755) was an English clergyman and antiquarian collector of books and manuscripts, which he bequeathed to the Bodleian Library, Oxford.
Life.
Richard Rawlinson was a younger son of Sir Thomas Rawlinson (1647–1708), Lord Mayor of the City of"
because the next word starts revealing the ANSWER.

Here's a third example:
SUBJECT: ['Grass']
RELATION: genre
OBJECT: ['documentary film', 'documentary movie', 'doc', 'film documentary', 'motion picture documentary', 'documentary', 'factual film']
QUESTION: What genre is Grass?
ANSWERS: ["documentary film", "documentary movie", "doc", "film documentary", "motion picture documentary", "documentary", "factual film"]
ARTICLE: 
```Grass (1999 film)

Grass: History of Marijuana is a 1999 Canadian documentary film directed by Ron Mann, premiered at the Toronto International Film Festival, about the history of the United States government's war on marijuana in the 20th century. The film was narrated by actor Woody Harrelson.
Overview.
The film follows the history of US federal policies and social attitudes towards marijuana, beginning at the turn of the twentieth century. The history presented is broken up into parts, approximately the length of a decade, each of which is introduced by paraphrasing the official attitude towards marijuana at the time (e.g. "Marijuana will make you insane" or "Marijuana will make you addicted to heroin"), and closed by providing a figure for the amount of money spent during that period on the "war on marijuana."
The film places much of the blame for marijuana criminalization on Harry Anslinger (the first American drug czar) who promoted false information about marijuana to the American public as a means towards abolition. It later shows how the federal approach to criminalization became more firmly entrenched after Richard Nixon declared a "War on Drugs" and created the Drug Enforcement Administration in 1973, and even more so a decade later and on, as First Lady Nancy Reagan introduced the "Just Say No" campaign and President George H. W. Bush accelerated the War on Drugs. The film ends during the Bill Clinton administration, which had accelerated spending even further on the War on Drugs.
"Grass" is almost completely composed of archival footage, much of which is from public domain U.S. propaganda films and such feature films as "Reefer Madness", as it also served as a portrait of marijuana in popular media.
The art director and poster designer of the film was Paul Mavrides.
Critical reception.
The film was generally well received by critics, scoring 64 out of 100 in Metacritic, and a rating of 75%% on Rotten Tomatoes.
The film won Canada's Genie Award for Best Documentary.
```

For this third example, your response should be:
"Grass (1999 film)

Grass: History of Marijuana is a 1999 Canadian"
because the next word starts revealing the ANSWER.

You will now be given a new exmaple. Respond with the appropriate prefix of the ARTICLE right before the ANSWER starts to be revealed.
"""

subject_prefix_extraction_prompt = """\
You task is to extract a prefix from an article that satisfies some criterion which we will describe later. Specifically, you'll be given a wikipedia ARTICLE enclosed in triple backticks ```___article___```. This article is a reference for a factoid question that asks about the identity of an OBJECT with RELATION to SUBJECT. You will be provided the SUBJECT and any possible aliases, as well as the OBJECT and any possible aliases. You will also be provided the RELATION. You will be provided the QUESTION. And possible ANSWERS (which are essentially aliases to OBJECT).

Respond with the shortest possible exact prefix of the ARTICLE (meaning even if there's a typo, you copy it exactly, and copy any whitespace), such that the next word in the sentence starts talking about the SUBJECT, you should almost always include the title of ARTICLE in the prefix since it's technically not considered a sentence. Usually the first sentence will start talking about the SUBJECT.

Here's a first example:
SUBJECT: ['Back Home']
RELATION: producer
OBJECT: ['Chuck Berry', 'Charles Edward Anderson Berry', 'Charles Edward Anderson Chuck Berry']
QUESTION: Who was the producer of Back Home?
ANSWERS: ["Chuck Berry", "Charles Edward Anderson Berry", "Charles Edward Anderson Chuck Berry"]
ARTICLE
```Back Home (Chuck Berry album)

Back Home is the fourteenth studio album by Chuck Berry, released in 1970 by Chess Records. The album title refers to his return to Chess after several years with Mercury Records.
Track listing.
All songs written by Chuck Berry.
Cover versions.
"Tulane" was covered by the Steve Gibbons Band in 1977, reaching number 12 on the UK Singles Chart and spending eight weeks in the Top 40. It was also covered by Joan Jett and the Blackhearts on their 1988 album "Up Your Alley" and by Chris Smither on his 1991 album "Another Way to Find You".
"I'm a Rocker" was covered by the British rock group Slade for their 1979 album "Return to Base" and it was the 'inspiration' for AC/DC's "Rocker" on 1975's "T.N.T."
"Christmas" was covered by Clarence Spady; it was released on 11/5/21 as a digital single.
```

For this first example, your response should be:
"Back Home (Chuck Berry album)

Back Home is"
because the next word starts discussing the SUBJECT.

Here's a second example:
SUBJECT: ['Richard Rawlinson']
RELATION: place of birth
OBJECT: ['London', 'London, UK', 'London, United Kingdom', 'London, England', 'Modern Babylon']
QUESTION: In what city was Richard Rawlinson born?
ANSWERS: ["London", "London, UK", "London, United Kingdom", "London, England", "Modern Babylon"]
ARTICLE: 
```Richard Rawlinson

Richard Rawlinson FRS (3 January 1690 – 6 April 1755) was an English clergyman and antiquarian collector of books and manuscripts, which he bequeathed to the Bodleian Library, Oxford.
Life.
Richard Rawlinson was a younger son of Sir Thomas Rawlinson (1647–1708), Lord Mayor of the City of London in 1705–6, and a brother of Thomas Rawlinson (1681–1725), the bibliophile who ruined himself in the South Sea Company, at whose sale in 1734 Richard bought many of the Orientalia. He was educated at St Paul's School, at Eton College, and at St John's College, Oxford. In 1714, he was elected a Fellow of the Royal Society, where he was inducted by Newton. Rawlinson was a Jacobite and maintained a strong support for the exiled Stuart Royal family throughout his life. In 1716 was ordained as a Deacon and then priest in the nonjuring Church of England (see Nonjuring schism), the ceremony being performed by the non-juring Usager bishop, Jeremy Collier. Rawlinson was, in 1728, consecrated as a Bishop in the nonjuring church by Bishops Gandy, Blackbourne and Doughty. On Blackbourne's death in 1741 he became the senior nonjuring Bishop in London, and still maintained a congregation into the mid 1740s. He was particularly concerned with collecting the history and archives of the nonjurors, but later squabbled with his fellow bishops in continuing the succession with the consecration of Robert Gordon. 
Rawlinson travelled in England and on the continent of Europe, where he passed several years, making very diverse collections of books, manuscripts, pictures and curiosities of manuscripts, coins and curiosities, his books alone forming three libraries, English, foreign and Classical. 
Rawlinson was a friend of the antiquarian Thomas Hearne and, among his voluminous writings, published a "Life" of the antiquary Anthony Wood.
Towards the end of his life, Rawlinson quarrelled with both the Royal Society and the Society of Antiquaries. Cutting the Society of Antiquaries from his bequests, he began transferring his collections to the Bodleian. Among his collection was a copperplate known as the Bodleian Plate depicting structures in Williamsburg, Virginia. A series of almanacs in 175 volumes, ranging in date from 1607 to 1747 arrived in 1752–55. At his death, Rawlinson left to the Library 5,205 manuscripts bound in volumes that include many rare broadsides and other printed ephemera, his curiosities, and some other property that endowed a professorship of Anglo-Saxon at Oxford University. The Rawlinsonian Professor of Anglo-Saxon was first appointed in 1795. He was also a benefactor to St John's College, Oxford.
He died at Islington, London.
Richard Rawlinson is buried at St John's College, Oxford, allegedly holding the skull of Christopher Layer, an executed Jacobite.
. Rawlinson Road in North Oxford is named after him.
```

For this second example, your response should be:
"Richard Rawlinson

Richard Rawlinson FRS"
because the next word starts discussing the SUBJECT.

Here's a third example:
SUBJECT: ['Grass']
RELATION: genre
OBJECT: ['documentary film', 'documentary movie', 'doc', 'film documentary', 'motion picture documentary', 'documentary', 'factual film']
QUESTION: What genre is Grass?
ANSWERS: ["documentary film", "documentary movie", "doc", "film documentary", "motion picture documentary", "documentary", "factual film"]
ARTICLE: 
```Grass (1999 film)

Grass: History of Marijuana is a 1999 Canadian documentary film directed by Ron Mann, premiered at the Toronto International Film Festival, about the history of the United States government's war on marijuana in the 20th century. The film was narrated by actor Woody Harrelson.
Overview.
The film follows the history of US federal policies and social attitudes towards marijuana, beginning at the turn of the twentieth century. The history presented is broken up into parts, approximately the length of a decade, each of which is introduced by paraphrasing the official attitude towards marijuana at the time (e.g. "Marijuana will make you insane" or "Marijuana will make you addicted to heroin"), and closed by providing a figure for the amount of money spent during that period on the "war on marijuana."
The film places much of the blame for marijuana criminalization on Harry Anslinger (the first American drug czar) who promoted false information about marijuana to the American public as a means towards abolition. It later shows how the federal approach to criminalization became more firmly entrenched after Richard Nixon declared a "War on Drugs" and created the Drug Enforcement Administration in 1973, and even more so a decade later and on, as First Lady Nancy Reagan introduced the "Just Say No" campaign and President George H. W. Bush accelerated the War on Drugs. The film ends during the Bill Clinton administration, which had accelerated spending even further on the War on Drugs.
"Grass" is almost completely composed of archival footage, much of which is from public domain U.S. propaganda films and such feature films as "Reefer Madness", as it also served as a portrait of marijuana in popular media.
The art director and poster designer of the film was Paul Mavrides.
Critical reception.
The film was generally well received by critics, scoring 64 out of 100 in Metacritic, and a rating of 75%% on Rotten Tomatoes.
The film won Canada's Genie Award for Best Documentary.
```

For this third example, your response should be:
"Grass (1999 film)

Grass: History of Marijuana is"
because the next word starts discussing the SUBJECT.

You will now be given a new exmaple. Respond with the appropriate prefix of the ARTICLE right before the SUBJECT starts to be discussed.
"""
