"""Build the labelled synthetic eval corpus (deterministic, seed 42).

Design
------
8 people x 6 attributes x 3 records = 144 records, plus 16 lexical-trap
records that share keywords with a *different* person's attribute.
Each record has a stable ``key`` (``<person>-<attribute>-<n>``) and a
``group`` (the person) so the eval can chain same-person edges.

Queries (24 total, gold = the 3 records of the queried attribute):
- 8 keyword queries: words taken straight from the records (BM25-friendly).
- 8 paraphrase queries: same meaning, different words — the entity name
  is shared but the content words are synonyms (vector-friendly).
- 8 trap queries: wording overlaps a *different* person's records
  (e.g. asking about Tom's baking when Maya is the baker); BM25 is
  expected to be fooled, the hybrid should not be.

Everything is generated from the templates below, so the gold sets are
true by construction. This corpus is SYNTHETIC — it measures relative
behaviour (hybrid vs single legs), not real-world accuracy. Real
numbers come from the LongMemEval-S run.
"""

from __future__ import annotations

import json
import os
import random

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "eval_data")

# (attribute, [3 record templates], keyword query, paraphrase query)
PEOPLE = {
    "maya": ("baker", [
        ("starter", [
            "Maya feeds her sourdough starter with rye flour every morning at six.",
            "The starter Maya keeps in the pantry bubbled over after four days without feeding.",
            "Maya's starter, begun in 2019, now leavens twelve loaves a week.",
        ], "Maya sourdough starter feeding",
         "how does Maya maintain her bread culture"),
        ("delivery", [
            "Maya delivered eight croissants to the corner cafe on Tuesday morning.",
            "The bakery van Maya uses broke down, so she carried the focaccia delivery on foot.",
            "Maya's Friday delivery route covers three cafes and the farmers market.",
        ], "Maya croissant delivery cafe",
         "who brings baked goods to the cafes on Maya's route"),
        ("oven", [
            "Maya's deck oven runs at 240 degrees for the morning bake.",
            "The oven thermostat Maya relies on drifts ten degrees high by noon.",
            "Maya preheats the stone deck for an hour before the first loaves go in.",
        ], "Maya deck oven temperature bake",
         "what heat does Maya's bread oven reach"),
        ("flour", [
            "Maya buys stone-milled flour from a mill in the valley, forty kilos a month.",
            "The flour Maya orders arrives in paper sacks that tear if they get damp.",
            "Maya switched to a coarser flour grind and the crumb opened up nicely.",
        ], "Maya stone-milled flour order",
         "where does Maya source her grain for bread"),
        ("market", [
            "Maya's market stall sells out of cinnamon rolls by nine on Saturdays.",
            "The stall Maya runs takes card payments now, after years of cash only.",
            "Maya sets up her market table at dawn and packs it down by one.",
        ], "Maya market stall cinnamon rolls Saturday",
         "when does Maya's weekend stall run out of pastries"),
        ("recipe", [
            "Maya's walnut loaf recipe uses seventy percent hydration and an overnight ferment.",
            "The recipe Maya wrote for rye bread calls for a full day of slow rising.",
            "Maya keeps her recipes in a stained notebook on the shelf above the mixer.",
        ], "Maya walnut loaf recipe hydration",
         "what is in Maya's notebook for her rye bread"),
    ]),
    "dev": ("cyclist", [
        ("bike", [
            "Dev's road bike has a cracked chainstay that needs welding before spring.",
            "The bike Dev rides to work has tubeless tyres that seal small punctures.",
            "Dev fitted a new saddle to his bike and the numbness on long rides stopped.",
        ], "Dev road bike chainstay cracked",
         "what is wrong with Dev's bicycle frame"),
        ("route", [
            "Dev's favourite training route climbs the ridge road and descends past the reservoir.",
            "The route Dev takes to work follows the river path for six kilometres.",
            "Dev mapped a new route that avoids the cobbled high street entirely.",
        ], "Dev training route ridge road reservoir",
         "which way does Dev ride when he trains"),
        ("race", [
            "Dev finished the spring criterium in twelfth place after a crash in the second lap.",
            "The race Dev entered next month is a hundred-kilometre road event with two climbs.",
            "Dev's race number for the weekend event is forty-seven.",
        ], "Dev spring criterium twelfth place",
         "how did Dev do in his last cycling competition"),
        ("repair", [
            "Dev trues his wheels with a spoke key and a zip tie as a gauge.",
            "The repair stand Dev bought folds flat and fits under the stairs.",
            "Dev replaced the bottom bracket on his winter bike in an evening.",
        ], "Dev trues wheels spoke key repair",
         "how does Dev fix a wobbly wheel"),
        ("gear", [
            "Dev's rain jacket packs into its own pocket and weighs almost nothing.",
            "The gear Dev carries on long rides includes a pump, two tubes and a multitool.",
            "Dev bought overshoes that keep his feet dry in the winter rain.",
        ], "Dev rain jacket pump tubes multitool gear",
         "what kit does Dev take on a long ride"),
        ("training", [
            "Dev trains four mornings a week, with intervals on Tuesdays and Thursdays.",
            "The training plan Dev follows builds to a peak in late June.",
            "Dev logs every training ride in a notebook with distance and average speed.",
        ], "Dev trains four mornings intervals Tuesday Thursday",
         "what is Dev's weekly workout schedule"),
    ]),
    "priya": ("gardener", [
        ("tomatoes", [
            "Priya's tomatoes ripened three weeks early after the hot June.",
            "The tomato plants Priya grows are a heritage variety with striped skins.",
            "Priya staked her tomatoes with bamboo canes tied in a figure eight.",
        ], "Priya tomatoes heritage variety ripened",
         "what kind of fruit is Priya growing this year"),
        ("compost", [
            "Priya turns her compost heap every Sunday and it steams in the cold.",
            "The compost Priya makes from kitchen scraps feeds the whole vegetable bed.",
            "Priya's compost bin smells sweet when the balance of greens and browns is right.",
        ], "Priya compost heap kitchen scraps",
         "how does Priya recycle garden waste"),
        ("watering", [
            "Priya waters the beds at dawn so the leaves dry before the sun is high.",
            "The watering can Priya uses holds ten litres and leaks at the seam.",
            "Priya set up drip lines that water the tomatoes while she is away.",
        ], "Priya waters beds dawn drip lines",
         "when does Priya irrigate her plants"),
        ("seeds", [
            "Priya starts her seeds indoors in March under a grow light.",
            "The seed packets Priya saves are labelled with the year and the variety.",
            "Priya swaps seeds with neighbours at the spring seed fair.",
        ], "Priya starts seeds indoors March grow light",
         "how does Priya begin her planting season"),
        ("pests", [
            "Priya picks slugs off the lettuce by torchlight after rain.",
            "The pests Priya fights most are aphids on the broad beans.",
            "Priya planted marigolds to keep whitefly away from the tomatoes.",
        ], "Priya slugs lettuce aphids broad beans pests",
         "what bugs attack Priya's vegetables"),
        ("harvest", [
            "Priya's harvest this year filled four crates of courgettes and beans.",
            "The harvest festival Priya attends is in the village hall every October.",
            "Priya freezes her harvest in batches labelled by month.",
        ], "Priya harvest crates courgettes beans October",
         "what did Priya gather from her garden"),
    ]),
    "omar": ("photographer", [
        ("camera", [
            "Omar's camera has a sticky shutter button that misses every tenth frame.",
            "The camera Omar shoots with is a rangefinder he bought second hand.",
            "Omar keeps his camera in a canvas bag with two spare batteries.",
        ], "Omar camera sticky shutter rangefinder",
         "what is wrong with Omar's picture-taking device"),
        ("lens", [
            "Omar's favourite lens is a fifty millimetre prime with a scratched front element.",
            "The lens Omar uses for portraits opens to f1.4 and melts the background.",
            "Omar cleaned fungus from an old lens with a cotton swab and patience.",
        ], "Omar fifty millimetre lens portraits",
         "which glass does Omar prefer for faces"),
        ("shoot", [
            "Omar's dawn shoot at the harbour caught the fishing boats in fog.",
            "The shoot Omar did for the bakery took six hours and forty rolls of film.",
            "Omar scheduled a shoot with the dance troupe for next Thursday.",
        ], "Omar dawn shoot harbour fishing boats",
         "where did Omar take pictures at sunrise"),
        ("prints", [
            "Omar prints his best work on matte cotton paper in the darkroom.",
            "The prints Omar made for the gallery show are all framed in black ash.",
            "Omar's print order for the wedding album took three weeks to finish.",
        ], "Omar prints matte cotton paper darkroom",
         "how does Omar produce his photographs on paper"),
        ("studio", [
            "Omar's studio has a north-facing window that gives soft light all day.",
            "The studio Omar rents shares a darkroom with two other photographers.",
            "Omar painted the studio walls grey to stop colour casts in portraits.",
        ], "Omar studio north-facing window portraits",
         "what is Omar's workspace like"),
        ("editing", [
            "Omar edits his photos in the evening, culling ruthlessly before breakfast.",
            "The editing preset Omar built warms the shadows and cools the highlights.",
            "Omar backs up his edited files to two drives after every shoot.",
        ], "Omar edits photos evening preset shadows",
         "when does Omar process his pictures"),
    ]),
    "lena": ("teacher", [
        ("classroom", [
            "Lena's classroom has a reading corner with cushions and a low shelf of picture books.",
            "The classroom Lena teaches in faces the playground and gets loud at break.",
            "Lena rearranged her classroom desks into groups of four for the new term.",
        ], "Lena classroom reading corner desks",
         "how is Lena's teaching room arranged"),
        ("lesson", [
            "Lena's lesson on fractions uses pizza slices and the children love it.",
            "The lesson Lena planned for Friday is a science experiment with magnets.",
            "Lena writes her lesson notes in pencil so she can change them on the fly.",
        ], "Lena lesson fractions pizza science magnets",
         "what does Lena teach with food examples"),
        ("exam", [
            "Lena's exam papers are marked by Friday and the results go home on Monday.",
            "The exam Lena set last term had a question everyone got wrong about tides.",
            "Lena keeps past exam papers in a filing cabinet sorted by year.",
        ], "Lena exam papers marked Friday results",
         "when do Lena's pupils get their test marks"),
        ("students", [
            "Lena's students planted a herb garden by the school gate this spring.",
            "The students Lena teaches wrote letters to the mayor about the library.",
            "Lena knows all thirty of her students by name within the first week.",
        ], "Lena students herb garden mayor library",
         "what did Lena's class do outside school"),
        ("books", [
            "Lena's book cupboard holds forty novels sorted by reading level.",
            "The books Lena recommends most are the ones with maps in the front.",
            "Lena reads a chapter aloud every afternoon before home time.",
        ], "Lena books cupboard novels reading level",
         "what does Lena read to her class"),
        ("trip", [
            "Lena's field trip to the museum is booked for the second week of May.",
            "The trip Lena organised last year was a farm visit with lambs and tractors.",
            "Lena counts heads three times before the coach leaves on any trip.",
        ], "Lena field trip museum May farm",
         "where is Lena taking her pupils"),
    ]),
    "sam": ("plumber", [
        ("pipes", [
            "Sam replaced the lead pipes in the old cottage with copper last winter.",
            "The pipes Sam fitted under the sink leak at the joint when the water is hot.",
            "Sam carries pipe in three sizes on the van roof rack.",
        ], "Sam replaced lead pipes copper cottage",
         "what plumbing did Sam renew in the old house"),
        ("boiler", [
            "Sam services boilers every autumn before the cold sets in.",
            "The boiler Sam installed has a ten-year warranty and a magnetic filter.",
            "Sam bled the radiators after fitting the new boiler and the house warmed evenly.",
        ], "Sam services boilers autumn warranty",
         "when does Sam maintain heating systems"),
        ("leak", [
            "Sam found the leak behind the bath panel with a tissue and a torch.",
            "The leak Sam fixed on Tuesday had been dripping into the ceiling for a month.",
            "Sam carries a moisture meter to find leaks without opening walls.",
        ], "Sam found leak bath panel ceiling",
         "how does Sam locate hidden water damage"),
        ("tools", [
            "Sam's pipe wrench is older than his van and still grips perfectly.",
            "The tools Sam keeps in his belt are a wrench, a cutter and a torch.",
            "Sam bought a press-fit tool that joins copper pipe in seconds.",
        ], "Sam pipe wrench cutter press-fit tools",
         "what equipment does Sam carry on jobs"),
        ("client", [
            "Sam's client on Oak Road wants the bathroom moved to the back bedroom.",
            "The client Sam visited yesterday has a tap that sings when it runs.",
            "Sam writes his client quotes on carbon paper so both sides keep a copy.",
        ], "Sam client Oak Road bathroom quote",
         "who does Sam do plumbing work for"),
        ("invoice", [
            "Sam's invoices are due in fourteen days and he chases them by phone.",
            "The invoice Sam sent for the boiler job lists parts and labour separately.",
            "Sam numbers his invoices by year so the accounts stay tidy.",
        ], "Sam invoices due fourteen days boiler",
         "how does Sam bill for his work"),
    ]),
    "nina": ("runner", [
        ("shoes", [
            "Nina's racing shoes have a carbon plate and weigh less than two hundred grams.",
            "The shoes Nina trains in are a size larger to leave room for swelling.",
            "Nina retires her shoes after five hundred kilometres, whatever they look like.",
        ], "Nina racing shoes carbon plate kilometres",
         "what footwear does Nina compete in"),
        ("marathon", [
            "Nina's marathon personal best is three hours twelve minutes, set in Berlin.",
            "The marathon Nina is training for starts at dawn on the first Sunday in October.",
            "Nina ran her first marathon in the rain and swore never again, then signed up twice more.",
        ], "Nina marathon personal best Berlin three hours",
         "what is Nina's fastest long race time"),
        ("route", [
            "Nina's favourite running route loops the lake and climbs the hill by the pines.",
            "The route Nina takes in winter avoids the icy towpath entirely.",
            "Nina measured her route with a wheel and it is exactly eight kilometres.",
        ], "Nina running route lake hill pines",
         "where does Nina go for her runs"),
        ("pace", [
            "Nina holds a five-minute kilometre pace for the first half of her long run.",
            "The pace Nina targets in training is ten seconds slower than race pace.",
            "Nina's watch beeps when her pace drifts above the target on hills.",
        ], "Nina five-minute kilometre pace watch",
         "how fast does Nina run"),
        ("injury", [
            "Nina's knee flared up after the downhill race and she rested for two weeks.",
            "The injury Nina worries about most is her Achilles, which tightens in the cold.",
            "Nina ices her knee after every run longer than an hour.",
        ], "Nina knee injury Achilles rest",
         "what hurts Nina when she runs"),
        ("nutrition", [
            "Nina eats porridge with banana two hours before a long run.",
            "The nutrition plan Nina follows adds salt tablets on hot days.",
            "Nina carries gels in her belt and takes one every forty minutes.",
        ], "Nina porridge banana gels nutrition run",
         "what does Nina eat before racing"),
    ]),
    "tom": ("chef", [
        ("menu", [
            "Tom's autumn menu leads with roast squash soup and a sage butter sauce.",
            "The menu Tom writes changes with the market, sometimes twice a week.",
            "Tom prints his menu on thick card that survives the steam from the pass.",
        ], "Tom autumn menu roast squash soup",
         "what dishes does Tom serve this season"),
        ("knives", [
            "Tom's knives are sharpened on a whetstone every Sunday without fail.",
            "The knives Tom uses are Japanese steel with handles of dark walnut.",
            "Tom keeps his knives in a roll that travels with him between kitchens.",
        ], "Tom knives whetstone Japanese steel",
         "how does Tom keep his blades sharp"),
        ("supplier", [
            "Tom's fish supplier calls at five in the morning with the day's catch.",
            "The supplier Tom trusts most is a farm that delivers vegetables at dawn.",
            "Tom changed supplier after three late deliveries in a single month.",
        ], "Tom fish supplier five morning vegetables",
         "who delivers Tom's ingredients"),
        ("reservation", [
            "Tom's restaurant takes reservations by phone only, no online booking.",
            "The reservation book Tom uses is leather-bound and lives by the till.",
            "Tom holds the corner table for regulars on Friday nights.",
        ], "Tom restaurant reservations phone book",
         "how do people book a table at Tom's place"),
        ("dessert", [
            "Tom's chocolate dessert is baked to order and takes twelve minutes.",
            "The dessert Tom is proudest of is a lemon tart with torched meringue.",
            "Tom serves his dessert with a spoon of clotted cream and nothing else.",
        ], "Tom chocolate dessert lemon tart meringue",
         "what sweet does Tom make best"),
        ("wine", [
            "Tom's wine list is short, local, and changes with the seasons.",
            "The wine Tom recommends with the squash soup is a dry white from the valley.",
            "Tom stores his wine in a cellar that stays cool all year.",
        ], "Tom wine list dry white squash soup",
         "what does Tom pour with dinner"),
    ]),
}

# Trap records: same keywords as another person's attribute, wrong person.
TRAPS = [
    ("maya", "Tom baked twelve loaves for the village fete and sold out by noon."),
    ("maya", "The bread Tom makes at home uses a bread machine, not a starter."),
    ("dev", "Lena rides her bicycle to school every day, rain or shine."),
    ("dev", "The cycling club Dev joined meets at the cafe Tom supplies."),
    ("priya", "Omar photographed Priya's garden for the local magazine last spring."),
    ("priya", "The tomatoes in Omar's studio window ripened in the sun."),
    ("omar", "Dev asked Omar to photograph his new bike for the race entry."),
    ("omar", "The camera shop Omar visits is next to the bakery Maya supplies."),
    ("lena", "Priya's daughter is in Lena's class and loves the reading corner."),
    ("lena", "The school garden Lena tends is planted with tomatoes Priya grew."),
    ("sam", "Tom called Sam to fix the leaking tap in the restaurant kitchen."),
    ("sam", "The boiler in Tom's restaurant broke down on the busiest night."),
    ("nina", "Dev runs with Nina on Sundays along the lake route."),
    ("nina", "Nina's running club meets at the cafe where Maya delivers."),
    ("tom", "Maya supplied the bread for Tom's restaurant opening night."),
    ("tom", "Tom taught Maya his recipe for walnut loaf at the market."),
]

# Trap queries: wording overlaps the wrong person's records; gold is the right person.
TRAP_QUERIES = [
    ("What did Tom bake for the village fete?", ["tom-dessert-1", "tom-dessert-2", "tom-dessert-3"]),
    ("Who supplies bread to Tom's restaurant?", ["maya-delivery-1", "maya-delivery-2", "maya-delivery-3"]),
    ("What is in the school garden?", ["lena-students-1", "lena-students-2", "lena-students-3"]),
    ("Who fixed the tap in the restaurant kitchen?", ["sam-leak-1", "sam-leak-2", "sam-leak-3"]),
    ("Where does the running club meet on Sundays?", ["nina-route-1", "nina-route-2", "nina-route-3"]),
    ("Who photographed the garden for the magazine?", ["omar-shoot-1", "omar-shoot-2", "omar-shoot-3"]),
    ("What does the cycling club do at the cafe?", ["dev-route-1", "dev-route-2", "dev-route-3"]),
    ("Whose tomatoes grow in the studio window?", ["priya-tomatoes-1", "priya-tomatoes-2", "priya-tomatoes-3"]),
]


def main() -> None:
    rng = random.Random(42)
    records: list[dict] = []
    queries: list[dict] = []

    for person, (_domain, attrs) in PEOPLE.items():
        for attr, templates, kw_q, para_q in attrs:
            keys = []
            for i, text in enumerate(templates, start=1):
                key = "%s-%s-%d" % (person, attr, i)
                records.append(
                    {"key": key, "group": person, "text": text, "type": "fact"}
                )
                keys.append(key)
            queries.append(
                {"question": kw_q, "gold_keys": keys, "kind": "keyword"}
            )
            queries.append(
                {"question": para_q, "gold_keys": keys, "kind": "paraphrase"}
            )

    # Trap records get their own keys but belong to no query's gold set
    # (they are pure distractors); their group is the person named in them.
    for i, (person, text) in enumerate(TRAPS, start=1):
        records.append(
            {"key": "trap-%d" % i, "group": person, "text": text, "type": "fact"}
        )

    for question, gold_keys in TRAP_QUERIES:
        queries.append(
            {"question": question, "gold_keys": gold_keys, "kind": "trap"}
        )

    rng.shuffle(records)
    rng.shuffle(queries)

    os.makedirs(OUT_DIR, exist_ok=True)
    corpus_path = os.path.join(OUT_DIR, "synthetic_corpus.jsonl")
    queries_path = os.path.join(OUT_DIR, "synthetic_queries.json")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2)

    kinds = {}
    for q in queries:
        kinds[q["kind"]] = kinds.get(q["kind"], 0) + 1
    print(
        "wrote %d records and %d queries (%s) to %s"
        % (len(records), len(queries), kinds, OUT_DIR)
    )


if __name__ == "__main__":
    main()
