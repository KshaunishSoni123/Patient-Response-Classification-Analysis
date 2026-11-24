# Sentiment annotation guidelines (real test messages)

Label the **emotional tone the patient expresses in the words of the message**, not the topic
and not whether the request is good news for the practice. This is the meaning the routing
policy assumes: *negative* routes a message to a human because the patient is unhappy,
*positive* insurance/payment messages get a thank-you.

## Decision procedure: ask in order, stop at the first "yes"

1. **Does the patient express thanks, relief, satisfaction, praise or excitement?** -> **positive** (`p`)
   Look for the words: "thank you so much", "that's great", "what a relief", "perfect", "works well".
2. **Does the patient express a complaint, frustration, worry, fear, distress or dissatisfaction?** -> **negative** (`n`)
   Look for the words: "too high", "still haven't", "ridiculous", "worried", "scared", "frustrated",
   "doesn't seem to cover", "denied again", "please help".
3. **Otherwise** -> **neutral** (`u`). Most patient messages are neutral: requests, questions,
   scheduling, giving information, and describing symptoms or facts.

A request is not positive just because it's routine or fulfillable. A symptom is not negative
just because it's a health problem. Only the patient's expressed feeling counts.

## Worked examples

| Message | Label | Why |
|---|---|---|
| "I need to book a dental appointment for a cleaning." | **neutral** | a request; no feeling expressed |
| "Can you help me find a dermatologist near me?" | **neutral** | a request |
| "Morning, around 9 AM would be great." | **neutral** | "great" is a scheduling preference, not satisfaction |
| "I have a slight rash on my arm." | **neutral** | a factual symptom description |
| "I'm having severe chest pain and shortness of breath." | **neutral** | severity words describe the symptom; the emergency rule handles urgency |
| "I'm really worried about these headaches." | **negative** | expressed worry |
| "My co-pay seems too high for this visit." | **negative** | complaint |
| "My insurance claim was denied." | **negative** | an unwanted outcome the patient is reporting as a problem |
| "My insurance doesn't seem to cover this medication." | **negative** | dissatisfaction or a problem with coverage |
| "What's the co-pay for specialist visits under my plan?" | **neutral** | an information question |
| "My member ID is INS987654321." | **neutral** | giving information |
| "That's great, thanks! Can I also ask about my lab results?" | **positive** | thanks and enthusiasm, then a neutral request; the feeling wins |
| "Thanks, but my bill is still wrong." | **negative** | mixed: label the part that should drive the response |
| "Could you please send me the forms? Thank you." | **neutral** | politeness alone is not positive |
| "I mean September 26th." | **neutral** | a short follow-up |

**Unsure after a few seconds?** Press `s` (skip). Skipped messages are excluded, not guessed.

## Process

- Run `python label_sentiment.py --annotator NAME --practice` once first: 15 made-up messages
  (not from the test set) with feedback, to calibrate.
- Then `python label_sentiment.py --annotator NAME`. It's blind (the model's prediction is never
  shown), resumable, and saves after every keypress.
- If a second person labels, they use their own `--annotator` name; Cohen's kappa is reported.

Keys: `n` negative · `u` neutral · `p` positive · `s` skip · `b` back · `g` show this file · `q` quit.

---

**History.** A first pass (`archive/real_test_sentiment_kshaunish_situation_based.csv`)
labelled the *situation* instead of the tone. Requests were labelled positive and health problems
negative (positive 170 / negative 155 / neutral 112), so it agreed with the sentiment head on
only 33% of messages. It also turned 50 insurance questions into `thank_you_message` reference
actions. It is archived and not used by the evaluation (only `labels/real_test_sentiment_*.csv`
at the top level is read).
