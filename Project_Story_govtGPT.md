# govtGPT ki kahani: ab tak kya kiya, kya socha, kya galat nikla

*Hindi + English mix. Ye ek story ki tarah likha hai, taaki interview me "maine ye kiya, ye socha, ye galat tha, phir ye check karke theek kiya" bol sako. Detail wali tables `Challenges_chunking_and_text_extraction.md` me hain.*

---

## Chapter 0: Goal kya tha?

Anil ko ek **production-grade RAG + agents system** banana tha, Indian IT/data law par:

- IT Act 2000
- IT Rules 2021
- DPDP Act 2023
- DPDP Rules 2025
- (Draft Digital India Bill sirf tab, aur `status: draft` ke saath, agar document mile)

Tools: **LangChain + LangGraph + Gemini API**. Seekhne ka tareeka: step by step. Anil pehle concept apne words me explain karta, phir main correct karta. Aur ek rule: **har step ke baad challenges aur unke solutions md file me likhne hain.**

---

## Chapter 1: LLM ki basics (Step 1)

Pehle bina RAG ke Gemini se seedha sawaal puche, taaki dikhe LLM akela kahan fail hota hai.

Seekha:
- **Token**: LLM words nahi, tokens padhta hai.
- **Temperature**: 0 karne se output stable hota hai, par **temperature 0 ≠ grounded**. Model ab bhi galat baat confidently bol sakta hai.
- **Hallucination**: Jo cheez model ko nahi pata, wo bhi wo likh deta hai. Law jaisi jagah par ye khatarnaak hai (jaise 66A ki baat, jo ab struck down hai).

Chhoti dikkatein:
- Gemini model ke naam par 404/503/429 errors aaye, model name aur retry se solve.
- `.env` ka format galat tha, theek kiya.
- Gemini 3 me jawab `.content` ki jagah `.text` se aata hai. Pehle `.content` use kiya, output ajeeb tha, phir `.text` par gaye.
- PowerShell me Hindi text ka encoding toot raha tha, UTF-8 set kiya.

**Lesson:** LLM akela law ke liye bharosemand nahi. Isliye RAG chahiye.

---

## Chapter 2: PDFs padhna aur chunking (Step 2)

Sabse zyada mehnat yahin lagi. PDF "text" jaisa dikhta hai, par asliyat me ganda hota hai.

### Jo socha tha
"pypdf se text nikalenge, 1000 char ke tukde karenge, ho gaya."

### Jo actually hua
1. **Hindi + English dono**: Gazette PDFs me Hindi bhi hai. DPDP Rules ka Hindi half garbled font me tha. **Fix:** jis page me Hindi akshar zyada hain use skip, aur Hindi lines hata di. Sirf English rakhi.
2. **Headers/footers noise**: "GAZETTE OF INDIA", page numbers, "REGD. No" har page par repeat ho rahe the. **Fix:** regex se noise lines hata di.
3. **Footnotes**: IT Act me neeche likha hota hai "2. Ins. by Act 7 of 2017..." Ye section ki tarah dikh raha tha. **Fix:** footnote regex.
4. **Table of Contents (ToC)**: ToC me bhi "43A. Compensation..." likha hota hai, to parser ne ToC ko asli section samajh liya. **Fix:** pehla section mil jaane ke baad hi parse karo, aur IT Act ke liye "—" dash zaroori.
5. **Amended sections** `3[43A. ...` jaise shuru hote hain (footnote number + bracket). Parser ye miss kar raha tha. **Fix:** heading regex me optional prefix.
6. **Section numbers sirf aage badhte hain**: ye idea kaam aaya. "Monotonic rule": agar abhi 43 par hain to agla 43A ya 44-48 ho sakta hai, "3." jaisa footnote nahi. Isse false headings reject ho gaye.

### Chunking ka decision
Section-aware chunking: pehle section/rule ke hisaab se todo (kyunki law me ek section ek poori baat hai). Sirf lambe section ko 1800 characters (~400-450 tokens) aur 200 overlap se todo.

Har chunk ke text ke andar **contextual header** daala: `[IT Act 2000 | Section 43A: Compensation...]`. Isse embedding ko bhi pata rehta hai ki chunk kahan se hai.

Metadata: act, short, year, doc_type, section, page, status, chunk_index...

### Legal status ka metadata
- **66A**: Supreme Court ne Shreya Singhal (2015) me strike down kiya, par PDF me abhi bhi chhapa hai. Isliye `status: struck_down`.
- **43A**: DPDP Act s.44(2) isse hatata hai, isliye `status: superseded`.

Bina iske system bol deta "66A ke under 3 saal ki saza", jo ab galat hai.

**Lesson:** Data cleaning RAG ka 60% kaam hai.

---

## Chapter 3: Embeddings aur vector DB (Step 3)

Pehle humne baat ki **embedding model kaise chunein**:
quality *apne data par*, language, max input length, dimension, cost/rate limit, latency, task type, hosting/privacy, ecosystem. Benchmark sirf shortlist banate hain. Model badla to **sab dobara embed karna padega**.

Chuna: `gemini-embedding-2` (3072 dims) + **Qdrant local mode**, cosine distance.

### Jo galat hua
- **588 points 298 chunks ke liye**: index dobara build karne par points double ho gaye. **Root cause:** random ids. **Fix:** stable id `uuid5(section|chunk_index|text)`, collection delete-recreate, aur assert ki ids unique hain.
- **429 rate limit**: Free tier me 100 embed requests/minute. **Fix:** batch 20, retry me lambi wait (30s, 60s...), aur query embeddings ka disk cache.
- Windows par shutdown ke time Qdrant/portalocker ka traceback aata tha. Ye sirf cleanup noise hai, kaam nahi bigadta.

Result: **298 chunks** (IT Act 137, IT Rules 48, DPDP Act 62, DPDP Rules 51).

---

## Chapter 4: Hybrid search ka argument (Step 4)

Anil ne pucha: normal vector search me kya issues hain, hybrid kyun?

### Jo maine socha (textbook)
Dense search "43A" jaise exact code miss kar sakta hai, isliye **BM25 + dense = hybrid**, aur dono ko **RRF** (Reciprocal Rank Fusion, score = Σ 1/(60+rank)) se jodo.

Anil ne apne words me samjhaya: "43A ka IDF zyada hota hai, 'section' har chunk me hai to IDF lagbhag 0. RRF dono se vote leta hai." Ye bilkul sahi tha.

Maine BM25 aur RRF explain kiya: TF saturation, IDF, length normalisation.

### Pehli galti: RRF me duplicate keys
Schedule ke saare chunks ka key "SCHEDULE" tha, to RRF me tak-raa gaye. **Fix:** key me `hash(page_content)` jod diya.

### Pehla naya sawaal: kya ye sach me kaam karta hai?
Hum sirf argument se maan rahe the. Isliye agla step evaluation tha.

---

## Chapter 5: Filters aur reranking (Step 5)

- **Metadata filters**: act, doc_type, status. Jaise `exclude_status=struck_down` se 66A hata sakte hain. Anil ne sahi kaha ki section code ko query se *filter* nahi karna chahiye, kyunki answer badal sakta hai.
- **LLM reranker**: Gemini se structured output me top candidates ko dobara rank karwaya. Anil ne bataya ki reranking sirf top 10-20 par karni chahiye, sab par nahi, kyunki neeche jaate jaate relevance dilute ho jaati hai. Ye bilkul sahi insight tha.

---

## Chapter 6: Evaluation: "kaam karta hai" ko number me badalna (Step 6)

30 sawaalon ka **gold set** banaya (`q, short, section, must` regex).

Metrics:
- **Hit@k**: kya sahi section top k me hai?
- **MRR** (Mean Reciprocal Rank): sahi jawab average kitne upar aaya. (Anil ne pehle MRR ko RRF se mila diya, maine correct kiya: dono alag cheezein hain. RRF *fusion* hai, MRR *metric* hai.)

Anil ka ek important sawaal: "agar section top 10 me nahi aaya to kya retrieval problem hai? hum to LLM ko top 10 de rahe hain." Haan: agar sahi chunk top-k me nahi hai, LLM ke paas jawab hi nahi hoga, wo ya to hallucinate karega ya refuse.

### Gold set khud galat nikla
Verify karte waqt **67B, 69, 69A** gold me the par parser ne unhe nikala hi nahi tha, aur DPDP s.16 ka regex match nahi hua. Yani evaluation se pehle hamara *data* galat tha.

**Root cause:** lambe titles do lines me wrap ho jaate hain:
```
67B. Punishment for abetment ... in
electronic form.– (1) Whoever
```
Dash agli line me hai, to parser ne heading reject kar di.

**Fix:** agar dash nahi mila to agli 2 lines me dekho. Result: IT Act 109 → **119 sections**. Ek parser bug ne gold set me 3 jawab gayab kar diye the.

### Pehla result (easy 30)
| | Hit@1 | MRR |
|---|---|---|
| dense | .87 | .933 |
| bm25 | .77 | .869 |
| hybrid | .90 | .950 |
| hybrid + LLM rerank | .83 | .911 |

Hybrid thoda jeeta, par **LLM reranker ne result kharab kiya** (.90 → .83). Reranker khud bhi galat ho sakta hai, isliye default me nahi lagaya.

---

## Chapter 7: Jab meri baat galat nikli (Step 7)

Mujhe laga easy set kaafi nahi, kyunki wo "khud se likhe hue" sawaal the. Isliye teen cheezein banayin:

1. **Hard set** (16 sawaal): paraphrase wale, bina section number ke, kai acceptable gold sections.
2. **Exact set** (10): jaha seedha "Section 43A" likha ho.
3. **Out-of-scope** (7): GDPR, Digital India Bill deepfake penalty, paneer butter masala, IPC murder, California CCPA, current PM, Aadhaar authentication.

### Jo maine socha tha
"Exact section lookups me BM25 chahiye, isliye hybrid dense se behtar hoga."

### Jo measure karne par mila
| Set | dense | bm25 | hybrid |
|---|---|---|---|
| Easy (30) Hit@1 / MRR | .87 / .933 | .77 / .869 | .90 / .950 |
| Hard (16) | .81 / .885 | .25 / .331 | **.44 / .640** |
| Exact (10) | **1.00 / 1.000** | - | .90 / .950 |

**Meri baat galat thi.**
- Dense search exact lookups par bhi perfect tha. Kyunki humne **chunk ke andar header** (`[IT Act 2000 | Section 43A ...]`) daala tha, to dense embedding ko bhi section number dikh raha tha.
- Hard set me **hybrid dense se bahut kharab** nikla. Wajah: BM25 kamzor tha (0.25) aur RRF ek kamzor list ko bhi barabar vote de deta hai, to kamzor list ne achhe list ko neeche kheench liya.

### Decision
Default retriever = **dense**. Hybrid code me rakha, par har naye corpus par dobara test hoga. Maine pichle step ka apna daava (BM25 "Section 43A" ke liye zaroori hai) file me **correction note** ke saath likh diya.

**Lesson (interview ke liye):** "Hybrid hamesha behtar hota hai" ek assumption hai. Humne apne data par measure kiya aur dense jeeta.

---

## Chapter 8: Jo jawab corpus me hai hi nahi

Retrieval **hamesha top-k laata hai**, chahe jawab corpus me ho ya na ho.

Maine cosine score ka **threshold** try kiya (0.72):
- Out-of-scope: **7/7 reject** ✔
- Lekin valid sawaal bhi: **9/46 reject** ✘

Yani sirf similarity score se "ye corpus me nahi hai" pakadna bharosemand nahi. Kuch valid sawaal ka score bhi kam aata hai.

**Result:** retrieval ke baad ek **grounded answer step** chahiye: LLM ko sirf retrieved chunks de, har claim par citation maange, aur agar chunks support nahi karte to "nahi mila" bole.

---

## Chapter 9: Jawab banana, bina jhooth bole (Step 8)

Ab tak sirf sahi chunks dhoondhna aata tha. User ko jawab chahiye, isliye Gemini se **open-book exam** karwaya: sirf 6 retrieved chunks do, har baat ke saath passage number aur exact quote maango, aur jawab na ho to "nahi mila" bolne do.

### Jo socha tha
"LLM ko prompt me sakht rules de denge (sirf chunks se jawab, citation do), to kaam ho jayega."

### Jo actually hua
1. **LLM ki citation par bharosa nahi kiya.** Pehle version me Gemini ne ek aisa "Fourth Schedule" cite kiya jo retrieve hi nahi hua tha, aur labels bhi alag-alag likhe ("Rule 3: (1)"). **Fix:** Gemini sirf *passage number* + *quote* de. Act aur section hamara code metadata se nikalta hai.
2. **Code se quote verify kiya.** Quote sach me us chunk me hai ya nahi, ye code dekhta hai. Agar saari citations fail hon to jawab ki jagah "not found".
3. **Mera verifier hi zyada sakht nikla.** Pehli verified run me 4 me se sirf 1 sahi sawaal ka jawab aaya (over-refusal). Teen wajah:
   - PDF ka spacing damage ("wit hin", "totake") aur Gemini ke saaf kiye hue words alag the.
   - "twenty-four" vs "twenty four".
   - Gemini ne do tukde "..." se jode the, to poora quote chunk me ek saath nahi tha.
   Maine pehle sochha ki Gemini jhooth bol raha hai, par debug karne par dikha ki *jawab sahi tha*, quote ka matching galat tha. **Fix:** matching me sirf letters/digits dekho, aur "..." se todkar har tukda alag check karo. Phir 4/4.
4. **Legal status LLM par nahi chhoda.** 66A ki warning (struck_down) metadata se code lagata hai.

### Result
- In-scope: **4/4** jawab mile.
- Out-of-scope: **7/7** "nahi mila".
- 66A ka jawab struck-down warning ke saath aaya.
- (Chhota test hai, benchmark nahi.)

### "Nahi mila" kaise tay hota hai
Teen halat me: (a) LLM ne `answerable=false` kaha, (b) jawab diya par citation nahi di, (c) citations di par code ne check kiya aur sab fail. Ye isliye ki LLM "haan jawab hai" bhi galat bol sakta hai, wahan code bachata hai.

### Anil ne apne words me kya samjhaya
Similarity score se "nahi mila" tay nahi hota, kyunki chunk similar ho sakta hai par wahi jawab nahi hota. Ye sahi tha. Maine ek baat jodi: score *vishay ki nazdeeki* batata hai, *jawab ka hona* nahi. Saboot: threshold 0.72 par 7/7 out-of-scope ruke par 9/46 sahi sawaal bhi ruke.

### Abhi bhi khula
- (Step 8b me kaafi had tak band hua: numbers + judge. Par test set chhota hai aur judge wahi model family hai.)
- Answer-level test set nahi hai (faithfulness, sahi refusal).
- PDF spacing damage ab bhi quotes ko tuta hua dikhata hai.

**Lesson:** Jo checker aap banate ho use *sahi* cases par bhi test karo, sirf galat cases par nahi. Mera verifier galat cases pakad raha tha par sahi cases bhi gira raha tha.

---

## Chapter 10: "Asli quote, galat jawab" (Step 8b)

Step 8 samjhate waqt Anil ne kaha: "hum check karte hain ki citation chunk me hai aur answer bhi chunk me hai." Maine dekha ki **answer wala hissa hamare code me tha hi nahi.** Code sirf quote ka hona check karta tha.

### Jo socha tha
"Quote verify ho gayi, matlab jawab bhi theek hoga."

### Jo check karne par mila
Maine jawab jaan-boojhkar bigade (par asli quote rehne di):
- T1: numbers badle (250 crore -> kuch aur)
- T2: kisi dusre sawaal ka jawab laga diya
- T3: matlab ulta kar diya ("must" -> "need not")

**Purana Step 8 verifier: 30 me se 0 pakda.** Meri soch galat thi.

### Fix: do nayi layers
- **L2, code:** jawab ke har number ko quote me dhoondho. "250 crore" aur "two hundred and fifty crore" ko ek integer banane ke liye chhota number-parser likha.
- **L3, LLM judge:** ek doosra call: "kya ye jawab in passages se poori tarah support hota hai?"

| Check | T1 number | T2 swap | T3 ulta matlab | Sahi jawab galti se roke |
|---|---|---|---|---|
| Sirf quote (purana) | 0/9 | 0/11 | 0/10 | 0/11 |
(T4 "galat number jo quote me hai": quote-only 0/6, numbers 0/6, judge 6/6)
| + numbers (code) | **9/9** | 8/11 | 0/10 | 1/11 |
| + LLM judge | 9/9 | **11/11** | 9/10 | 1/11 |

### Kya seekha
1. **Code numbers pakadta hai, matlab nahi.** "Must" ka "need not" ban jana code ko nahi dikhta, wahan judge chahiye.
2. **Ek sahi jawab bhi roka gaya.** Usme "five lakh" tha jo cited quote me nahi tha. Sach me uncited tha, isliye maine rule nahi dhila kiya. **(Correction, Step 13b: ye meri galti thi. Quote me "fiv e lakh" tha, yaani PDF spacing damage se number padha nahi gaya. Wo jawab sahi tha aur checker ne galat refuse kiya. Ab spacing repair ke baad accept hota hai.)**
3. **Experiment ki kamzori:** bigaade hue jawab maine khud banaye, aur sirf 11 sahi + 30 bigade. Number 9/9 isliye aasan tha. Judge aur generator ek hi model family hain, to ek jaisi andhi jagah ho sakti hai. Ek jawab (1/10) judge se bhi nikal gaya.
4. **Decision:** numbers check hamesha ON (muft aur exact). Judge sirf bhaari sawaalon par (penalty, deadline), kyunki har jawab par extra call lagti hai.

### T4: Anil ka sawaal (number check par bharosa kyun nahi)
Anil ne kaha number check sirf *presence* dekhta hai, ye nahi ki number *isi baat ka hai*. Maine isse test banaya: ek chunk me "24 hours" aur "15 days" dono hain, jawab me galat wala number, quote me dono. **Numbers check: 0/6. LLM judge: 6/6.** (Caveat: ye galat jawab bhi maine hi banaye, bade seedhe hain.) Matlab Anil sahi the: number check akela kaafi nahi, par pehli sasti layer ke roop me kaam ka hai.

**Lesson:** "Citation hai" ka matlab "jawab sahi hai" nahi. Verifier ko bigade hue cases par test karo, tabhi pata chalta hai wo kuch pakadta bhi hai ya nahi.

---

## Chapter 11: Agent banana (Step 9)

Ab tak pipeline fixed thi: retrieve, phir answer. Agent me LLM khud tay karta hai: kaun sa tool, kitni baar, kab rukna. Maine ek **explicit loop** likha (koi framework magic nahi) taaki har safety rail dikhe.

**Tools:** `search_law` (topic se dhoondho) aur `get_section` (exact section uthao).
**Safety (sab code me):** MAX_STEPS=6, same call dobara = block, tool exception = ERROR message jo LLM padh sake, aur agent sirf *evidence jama karta hai*. Final jawab wahi Step 8 ka grounded + verified pipeline likhta hai, to agent citation checks bypass nahi kar sakta.

### Jo socha tha
"Multi-part sawaal (IT Act 43A vs DPDP) ke liye agent zaroori hoga, kyunki do jagah search karni padegi."

### Jo dikha (5 chhote runs)
- **Exact lookup (66E):** sahi tool chuna, par wahi call dobara ki. Loop guard ne **sach me block kiya**, phir bhi 2 extra searches kiye.
- **Compare 43A vs DPDP:** agent 6 steps me MAX_STEPS tak pahunch gaya (13 chunks). **Par seedha one-shot RAG ne bhi dono sections (43A, s.44) pakad liye the.** Meri soch yahan sabit nahi hui: agent ki zaroorat dikhi nahi, bas zyada calls lagin.
- **Tool fail (2 baar):** program crash nahi hua, ERROR LLM ko mila. Par recovery kharab: agent ne **section numbers andaze se try kiye** (33, 34, 25, 32) aur jawab dhundhla raha ("penalty Schedule me hai", amount nahi).
- **Section 999 (exist nahi karta):** `get_section` ke error me valid sections ki list thi, agent ek call me ruk gaya.
- **Out-of-scope (CCPA):** 1 search, ruk gaya, refuse. Sahi.

### Kya seekha
1. Loop guard sirf theory nahi, asli run me chala.
2. Tool error ko "samjhane wala" likho (valid options batao), tabhi LLM recover kar paata hai.
3. **Agent hamesha behtar nahi.** Mujhe abhi ek bhi example nahi mila jisme one-shot RAG fail ho aur agent jeete. Agent = zyada cost + zyada variance, isliye pehle fail hone wala test banao, phir agent lagao.
4. Abhi memory nahi hai (har run khali se). State/memory/checkpoints LangGraph me aayenge.

---

### Anil ke jawab aur mere sudhaar (Step 9)
- **Agent vs plain RAG:** Anil: plain RAG ek seedha raasta hai, agent khud tay karta hai kaun sa tool, kitni baar. Sahi. Sudhaar: **LLM agent ka dimaag hai, tool nahi.** Tools (search, get_section) haath hain. Aur ek chakra hai: tool ka natija dekho, phir socho, phir karo.
- **Loop code me kyun:** Anil: prompt ko LLM chhod ya andekha kar sakta hai, isliye code me. Sahi. Sudhaar: prompt *nivedan* hai, code *tala*. Prompt me DONE kehna madad karta hai, par guarantee code ki step limit aur repeat block deta hai. Asli run me prompt ke bawajood agent ne wahi call dohrayi.
- Tale ki keemat: step limit aane par agent aadhe kaam par ruk sakta hai, isliye fallback: jitna evidence mila usi se Step 8 verifier ke saath jawab banao.

---

## Chapter 12: LangGraph (Step 10)

### LangGraph kyun chuna (aur kyun nahi)
Step 9 me maine apna loop khud likha. Kaam karta tha, par teen cheezein haath se banani padti thin: state, routing, memory. LangGraph wahi cheezein framework ke andar deta hai:

1. **State**: ek typed dict (question, queries, evidence, result, retries). Har node usse padhta/likhta hai. Ye "reducer" bhi deta hai (`history` me naye items *jud* jaate hain, replace nahi hote).
2. **Conditional edges**: "answer verify hua to finalize, nahi hua aur retry bacha hai to rewrite" ye ek jagah likha hai, graph ki shakal me dikhta hai, alag test ho sakta hai.
3. **Checkpointer + thread_id**: har node ke baad state save. Isse (a) chat memory, (b) crash ke baad resume, (c) human approval ke liye pause (interrupt) milta hai.
4. **recursion_limit**: framework level par loop cap (humara apna `retries` counter bhi hai).

**Jo LangGraph nahi karta:** retrieval behtar nahi banata, jawab sahi nahi banata. Safety wahi hai jo pehle thi: answer node = Step 8/8b verifier, koi node usse skip nahi kar sakta.

### Graph ki shakal
`plan` (in-scope? sawaal ko akela-khada banao, 1-3 search queries) -> `retrieve` -> `answer` (generate + verify) -> jawab mila to `finalize`; refuse hua aur retries bache to `rewrite` -> wapas `retrieve`. Out-of-scope sawaal `plan` se seedha `finalize` (retrieval hi nahi).

### Jo measure kiya (16 hard + 7 out-of-scope, one-shot vs graph)
| | one-shot | graph |
|---|---|---|
| Hard answered (16) | 13 | 13 |
| Answered aur gold section cite kiya | 12 | 13 |
| Out-of-scope refused (7) | 7/7 | 7/7 |

Ye **saaf jeet nahi hai.** Teen sach:
1. 6 sawaalon par dono ke jawab alag the, par dono taraf (kuch graph ne bacha, kuch graph ne gira diye). Pichli run me wahi sawaal ulte nikle the. Matlab temperature 0 par bhi `numbers` check wali refusals run-to-run hilti hain (13 vs 13 noise ke andar hai, 1 sawaal ka farak ~6 points).
2. **Self-correction loop ek baar bhi nahi chala.** Zyadatar refusals `numbers` se thin, aur maine retry sirf `llm`/`quote` refusal par rakha (kyunki naya search sirf tab madad karta hai jab chunk hi galat aaye). Yani asli failure retrieval ki nahi, generation/verifier ki hai, aur rewrite use nahi chhoo paata.
3. Graph ko ek jeet mili jo number me nahi dikhti: `plan` ne multi-part sawaal ko kai queries me toda aur evidence 12 chunks tak gaya, par ye bina LangGraph ke bhi ho sakta tha.

### Memory ka test (ye asli fayda nikla)
Turn 1: "DPDP Act me data breach ka penalty?" Turn 2: "Aur purane IT Act ne same cheez par kya kaha?" Same `thread_id` par graph ne "same cheez" ko samajh kar 43A (superseded) cite kiya. **Nayi thread par wahi sawaal** (bina history) ek bilkul alag jawab laya (electronic records ki legal recognition), verified aur cited, par sawaal ka jawab nahi. Yaani verifier "jawab chunk se support hota hai" check karta hai, "ye sawaal ka jawab hai" nahi, aur bina memory ke vague follow-up par system confidently off-topic bol deta hai.

### Kya seekha
1. LangGraph ka fayda **structure, memory, resume** hai, accuracy nahi. Hum "ye jeeta" nahi bol sakte, sirf "ye wahi accuracy deta hai aur memory + saaf routing deta hai".
2. Retry-edge tabhi kaam ka hai jab failure us edge ke theek karne laayak ho. Pehle failure ka type dekho (`refused_by`), phir edge banao.
3. Agla asli sudhaar: `numbers` refusal ko alag route do (jawab dobara generate karo, chhote evidence ke saath) aur ek aisa test set banao jahan one-shot sach me fail ho.
4. Human approval (interrupt) abhi nahi banaya: Q&A read-only hai, koi risky action nahi. Wo tab aayega jab system kuch bhejega/likhega (jaise notice draft).
5. Chhoti dikkat: checkpoint me `Result`/`Citation` dataclass ko LangGraph "unregistered type" warning deta hai (future me block). Production me allowed_msgpack_modules me register karna hoga.

---

## Chapter 13: `numbers` refusal ka sudhaar (Step 11), aur jo nahi chala

Step 10 me dikha tha ki retry-edge kabhi nahi chala, kyunki zyadatar refusals `numbers` check se the. To maine ek naya route banaya: `numbers` refusal par **wahi evidence, dobara generate**, aur model ko batao ki kaunsa number reject hua.

### Jo socha tha
"Feedback milne par model number theek kar dega, to refusals kam honge."

### Jo measure kiya (2 repeat x 16 hard sawaal = 32 runs)
| | Answered | Answered + gold cite |
|---|---|---|
| one-shot | 22/32 | 21/32 |
| graph, fix OFF | 28/32 | 26/32 |
| graph, fix ON | 31/32 | 28/32 |

**Meri soch sabit nahi hui.** Fix node poori run me sirf **1 baar** chala aur wo bhi fail hua. To OFF aur ON ka 28 vs 31 ka farak fix ki wajah se nahi, **noise** hai (~3 sawaal). Isse noise ka size bhi pata chala. Graph vs one-shot ka farak bada hai, par Step 10 me wahi 13 vs 13 tha, isliye ise "suggestive" hi kehna sahi hai.

Chhoti galti bhi pakdi: mera `retry()` wrapper daily-quota error nigal raha tha ("gave up" likh ke), asli wajah (Gemini free tier 500 requests/din) alag se run karke dikhi. Ab wrapper quota error par seedha ruk jaata hai aur wajah print karta hai.

### Debug: "maximum fine" sawaal kyun refuse hota hai
Same evidence se 6 baar generate kiya:
- 5/6: ek citation ("two hundred and fifty crore"), sab theek.
- 1/6: model ne poori penalty table likh di aur **15 aur 28** jaise numbers bhi likhe jo kisi quote me nahi the.

Yaani **jawab sahi tha, checker ne galat refuse kiya** (false refusal). Wahi Step 8 wali kahani: mera verifier zyada sakht. Isi se ye bhi samajh aaya ki fix node kaam kyun nahi aaya: feedback "number quote me hona chahiye" kehta hai, par model wahi table dobara likh deta hai.

### Kya seekha
1. Pehle failure ki **wajah** dekho, phir fix banao. Maine fix banaya, par asli dikkat checker ki thi.
2. Ek hi baar chalne wali cheez ka "benefit" nahi naap sakte (1 firing, 0 success).
3. Noise measure karna zaroori hai: identical graphs me 28 vs 31 aa gaya. Isse kam farak par kuch claim nahi karna.
4. **Agla verifier kaam:** numbers check me "section/rule N" ko ignore karna ya cited passage ke andar ke numbers allow karna. Par isse pehle Step 8b ke tampered set (T1-T4) par dobara test karna hoga, taaki galat numbers na ghusen.

---

## Chapter 14: API aur logging (Step 12)

Ab tak sab kuch scripts se chalta tha. Production jaisa banane ke liye ek **FastAPI** service banayi (`govtgpt/api.py`):

- `POST /ask` (question, thread_id) -> answer, citations, warnings, refused_by, request_id, latency.
- `GET /health`.
- Har request par **ek JSON log line** (`logs/requests.jsonl`): request_id, thread_id, question, queries, retrieved sections, refused_by, answerable, retries/fixes, latency, error. Isse baad me "kaun sa sawaal kyun refuse hua" dekha ja sakta hai (jaise upar wala numbers refusal).
- Guardrails: question length cap, khali question reject, LLM/quota error par 503 (crash nahi), aur "not found" jawab ko 200 ke saath saaf `answerable=false` (error nahi).

(Details aur test ke results neeche challenges file me.)

---

## Chapter 15: Numbers check ko sudhaara (Step 13)

Step 11 ke debug se pata chala tha ki "max fine" jawab sahi tha par numbers check ne **section references** ("section 15", "28") ko quantity samajh kar refuse kiya.

### Fix
`split_refs`: "section/rule/clause N" wale numbers alag nikalo. Unhe tab allow karo jab wo **cited passage** me kahin bhi ho. Amount aur duration ab bhi **quote ke andar** hi hone chahiye. Yaani wrong amount ab bhi pakda jaata hai.

### Test (offline, cached drafts, koi LLM call nahi)
- "max fine" jaisa jawab (refs 8/15/28): pehle refuse, **ab accept**. 8, 15, 28 teeno Schedule passage me mile, to jo wajah socha tha wo sahi nikli.
- Bigade hue jawab: T1 9/9, T2 8/11, T4 0/6 (judge chahiye), T5 (jhoothi section 999) 11/11, T6 (asli par cite nahi hui section) 11/11. **Pehle jitna hi**, kuch nahi bigda.
- Ek honest jawab ab bhi refuse hota hai ("five lakh" quote me nahi), wo sahi refusal hai.

### Seekha / caveat
1. Verifier loose karte waqt har baar wahi bigade-hue set dobara chalao, tabhi pata chalta hai ki kuch ghusa to nahi.
2. R1 mera banaya replica hai, asli failing output save nahi tha. Live model par 1-in-6 refusal rate dobara naapna baaki hai.
3. Regex ka bug: patch script ne `` ko backspace bana diya aur regex chup-chaap kuch match nahi karta tha. Naye regex ko examples par test karo.

---

## Chapter 16: Live measurement aur refine (Step 13b/c)

Maine "max fine" sawaal ko live 10 baar generate kiya (aur 11 aur sawaal 2-2 baar), har draft ko purane aur naye rule se verify kiya.

### Jo socha tha
"Step 13 ka reference rule baaki refusals bhi theek kar dega, 1-in-6 jaisa rate hoga."

### Jo mila
Asli false-refusal rate **3/10** tha (6/32 overall), aur Step 13 ne sirf 6 me se **1** theek kiya. Baaki 5 ka kaaran maine saved drafts padh kar dekha:
1. Model jawab ke andar `[2]` jaise passage markers likhta hai, checker ne "2" ko quantity samjha.
2. Quote me PDF spacing damage ("twohundred and fiftycrore"), parser padh nahi paaya.
3. "five lakh" waala jawab, jise maine Step 8b me "sach me uncited" kaha tha, quote me "fiv e lakh" ke roop me **tha**. Meri purani baat galat thi (correction Chapter 10 me jod di).

### Refine
`[N]` markers hatao, aur **sirf source text** par `repair_spacing` lagao ("fiv e" -> "five", "twohundred" -> "two hundred"). Naye rule ke baad 32 me se **0** numbers-refusals.

### Safety check
- Tamper set dobara: T1 9/9, T2 8/11, T5 11/11, T6 11/11, honest 0/11. Kuch bhi bigda hua jawab ab aasaani se pass nahi hota.
- Repair poore corpus (298 chunks) par: sirf **2 chunks** ke numbers badle (IT Act s.44 me 150000, s.67 me 500000), dono sahi. Yaani allowed set lagbhag nahi badha.
- T4 0/6 se 1/6 ho gaya, wajah mujhe samajh nahi aayi. Safe direction me hai par us par bharosa nahi karna.

### Seekha
1. Pehle jo maana ("1-in-6", "genuinely uncited") use dobara live data se check karo. Dono andaaze galat the.
2. Verifier ka har false refusal ek asli draft se aana chahiye, aur har change ke baad tamper set dobara.
3. Spacing damage sirf search nahi, verifier ko bhi tod deta hai. Asli fix ingestion me hai (A12), ye repair uska band-aid hai.

---

## Aaj ka stopping point (session 1 khatam)
Ho gaya: Step 1 se 9 (8b aur T4 ke saath). **Agla: Step 10 LangGraph** (StateGraph, state/memory, conditional routing, checkpoints, human approval, self-correction). Pehle ek aisa sawaal dhoondhna jisme one-shot RAG sach me fail ho, taaki agent/graph ka fayda naap sakein.

Baaki chhote kaam: requirements.txt me missing packages (pypdf, langchain-text-splitters, qdrant-client, langchain-qdrant, rank-bm25, langgraph), .gitignore me artifacts/qdrant_db/data_txt, PDF spacing damage aur DPDP U+FFFD fix, answer-level test set, alag model ko judge banana.

---

## Ab tak ki known limitations (honest list)

1. Words ke beech spacing damage (PDF extraction se).
2. DPDP Act me `�` characters aur galat auto-titles.
3. "Definitions" (jaise Section 2) retrieval me neeche aati hain (hybrid me rank 14).
4. Stopword removal nahi kiya.
5. IT Rules 2021 me baad ke amendments nahi hain.
6. DPDP Rules ka commencement phased hai.
7. Sirf English.
8. Eval sets chhote hain (30 / 16 / 10 / 7).
9. Act aur Rules ke neighbouring sections me confusion.

---

## Ab tak ke scorecard ka summary

- Step 1: LLM basics ✔
- Step 2: Ingestion + chunking (298 chunks) ✔
- Step 3: Embeddings + Qdrant ✔
- Step 4: BM25 + hybrid (RRF) ✔
- Step 5: Filters + LLM rerank ✔
- Step 6: Retrieval evaluation ✔
- Step 7: Hard + out-of-scope evaluation ✔
- Step 8: Grounded answers, verified citations, `struck_down` warnings, "not found" refusal ✔
- Step 8b: Faithfulness checks (numbers in code + LLM judge) on tampered answers ✔
- Step 9: Tool-calling agent (ReAct loop, loop guard, tool failure) ✔ (par agent ki zaroorat abhi sabit nahi)
- **Step 10 (agla):** LangGraph (StateGraph, routing, checkpoints, human approval, self-correction), aur production (API, logging, caching, guardrails).

---

## Teen baatein jo har interview me kaam aayengi

1. **Data pehle, model baad me.** Humare sabse bade bugs parser me the, model me nahi.
2. **Measure karo, maan mat lo.** Hybrid "theory" me jeeta, data par haara.
3. **Retriever kabhi "I don't know" nahi bolta.** Wo hamesha kuch laata hai, isliye generation step me grounding aur refusal zaroori hai.
4. **LLM ki citation par bharosa mat karo, code se verify karo** (aur verifier ko sahi cases par bhi test karo).
