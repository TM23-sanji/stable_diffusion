import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from huggingface_hub import login
from pathlib import Path
import json

login("YOUR_HF_TOKEN")

BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_MODEL_ID = "tm23hgf/anime-sdxl-lora"
OUTPUT_DIR    = Path("generated_100")
OUTPUT_DIR.mkdir(exist_ok=True)

PROMPTS = [
    # ── Ultra-detailed single characters ────
    "a teenage girl with asymmetric bob cut, one side dyed crimson, standing under a broken vending machine in a flooded subway tunnel, bioluminescent algae glowing on the walls, reflection in ankle-deep water, melancholic expression",
    "elderly swordsman, deep facial scars, torn haori, sitting on a collapsed torii gate after a battle, ravens circling above, ash falling like snow, distant burning village",
    "small boy in oversized raincoat holding a paper lantern, standing alone at a crossroads in a bamboo forest at 3am, thick fog, fireflies, no other light source",
    "tall woman in a crisp white suit, jet-black hair pulled back severely, standing at the edge of a helipad at dawn, briefcase in hand, city 40 floors below wrapped in smog",
    "a 10-year-old girl with dirt-smudged face, mismatched secondhand clothes, pressing her nose against a toy shop window, reflection showing her wide wondering eyes",
    "teenage boy, shaved head, prison jumpsuit, sitting cross-legged on a rooftop water tower at dusk, reading a stolen library book, pigeons around him",
    "a young woman mid-sneeze, messy bun half-falling out, flour on her cheek, apron covered in batter, chaotic home kitchen behind her, morning light",
    "muscular older woman, grey streaks in braids, calloused hands wrapping her own knuckles, underground boxing gym, single hanging bulb, smoke-stained walls",
    "23-year-old man in a soggy business suit, missing one shoe, sitting on a curb outside a closed convenience store at 2am, head in hands, briefcase abandoned beside him",
    "a girl around 16, sitting in the back of a moving pickup truck at night, legs dangling off the edge, head tilted back looking at stars, rural highway, wind-blown hair",

    # ── Unusual pairings & dynamics ─────────
    "yakuza boss in full sleeve tattoos tenderly feeding stray kittens in a back alley, bodyguards watching awkwardly at a distance, harsh neon overhead",
    "a little girl teaching her enormous grandfather how to use a smartphone, both hunched over the tiny screen, warm kitchen table, reading glasses on the old man",
    "two exhausted paramedics eating vending machine sandwiches on a hospital loading dock at 4am, still in uniform, not talking, just existing together",
    "rival gang leaders sharing an umbrella at a bus stop during a sudden downpour, both too proud to acknowledge the situation, tense silence",
    "a street artist caught mid-tag by a security guard who turns out to be his childhood art teacher, both frozen in shock, 3am warehouse district",
    "teenage girl teaching her robot how to cry, dimly lit bedroom, robot chest panel open revealing sparking internals, tissues on the floor",
    "a salaryman and a stray dog both staring at the same closed ramen shop sign at 11pm, rain, both equally devastated",
    "two old women, former rivals, playing shogi in a nursing home, neither willing to lose, other residents watching breathlessly",
    "a blind street musician and a deaf girl who learned to feel vibrations, her hand on his speaker, underground passage, commuters blurring past",
    "a ghost who does not know she is dead still going through her morning routine, brushing teeth, coffee machine running, winter light through frosted windows",

    # ── Environments without characters ─────
    "an abandoned love hotel in the japanese countryside, vines crawling through broken windows, faded pink wallpaper, heart-shaped mirror cracked on ceiling, golden afternoon shaft of light",
    "a capsule hotel at 3am viewed from above, each pod a different color of TV glow, one pod dark, one pod playing what sounds like crying",
    "a flooded supermarket, water up to the shelves, a shopping cart drifting past canned goods, emergency lighting flickering, no humans",
    "a shrine in dense cedar forest, paper wishes tied to every branch so many they block out the sky, grey morning light filtering through",
    "an internet cafe at dawn, half the booths occupied with sleeping figures hunched over keyboards, cigarette smoke hazing the monitor light",
    "a decrepit pachinko parlor mid-renovation, half the machines ripped out, one machine still lit playing its jingle to no one, construction dust everywhere",
    "a commuter train car at 6am, every seat taken by a salaryman in identical grey suit, all asleep in exactly the same position, eerie symmetry",
    "the moment just after a summer festival ends, lanterns going dark one by one, paper trash drifting across empty grounds, lone stall owner sweeping",
    "inside a typhoon shelter, dozens of fishing boats crowded together, ropes tangling, hull numbers in faded paint, dark churning water outside",
    "an izakaya at closing time, chef alone at the counter eating leftovers, every table showing the aftermath of the night, warm amber light",

    # ── High concept & surreal ───────────────
    "a girl whose shadow moves independently, her shadow currently trying to sneak out the door while she sleeps, moonlit room, both with different body language",
    "a world where memories are stored in glass jars on shelves, a janitor carefully dusting them in a vast library stretching to infinity",
    "a city that only exists at night, during the day it becomes a field of ordinary grass, the transition happening at 5:58am, buildings flickering",
    "a boy who collects only the last pages of discarded books, read thousands of endings but never a beginning, surrounded by last pages pinned to every wall",
    "a map that updates in real time based on collective human emotion, fear makes streets red, joy turns them gold, a cartographer watching it shift",
    "fish that swim through air instead of water, a fisherman with a net 30 feet above a busy street corner, pedestrians completely unfazed",
    "a lighthouse that shines inward instead of outward, illuminating the keeper's memories, visible from outside as shifting scenes through the glass",
    "a postal worker delivering letters to people who no longer exist, she knows, places them carefully at doorsteps of demolished buildings anyway",
    "time moves differently in this restaurant, customers age visibly during a long meal, the chef remains exactly the same, has been for decades",
    "a cartographer who maps places that do not exist yet, her maps keep coming true, she has stopped publishing them, one sits unfinished on her desk",

    # ── Grounded emotional moments ───────────
    "university student calling her parents for the first time after 6 months of silence, sitting on dorm room floor, phone to ear, not sure if they will pick up",
    "a man cleaning out his late mother's apartment, pausing over a photo he has never seen, sitting among boxes, late afternoon light",
    "two brothers after a funeral, one crying openly, one completely still, they have not spoken in years, crematorium parking lot, winter",
    "a teacher staying late to correct papers, one student essay describes a home life in crisis, she has stopped marking and is just reading, deeply concerned",
    "a young woman in a hospital waiting room, coat still on, hands clasped, staring at nothing, it has been four hours, vending machine humming behind her",
    "a retired athlete watching his replacement win gold on a small bar TV, complex expression somewhere between proud, sad, and relieved",
    "a girl on her first day at a new school eating lunch alone by a window, pretending to be fascinated by something outside, holding it together",
    "two teenagers who just broke up sitting at opposite ends of the same park bench, too proud to leave first, the city getting dark around them",
    "a chef receiving a one-star review on his phone, stepping out the restaurant back door, standing in the alley for a moment, cigarette unlit in his hand",
    "a mother watching her son board a train overseas, already texting him even though he is still visible through the platform window",

    # ── Action without cliche ────────────────
    "a pickpocket mid-theft on a crowded train, her hand in a coat pocket, but she has just found a suicide note, frozen, reading it, target unaware",
    "a firefighter emerging from a burning building carrying a box of irreplaceable documents, not a person, just this box, exhaustion and decision on her face",
    "a motorcycle courier weaving through gridlocked traffic during a typhoon warning, last delivery of the night, the address is a hospital",
    "a woman defusing a bomb while arguing on the phone with her landlord about her broken heater, completely compartmentalizing both crises",
    "two snipers on opposing rooftops, one has just lowered his rifle because he recognizes her from high school, both frozen, long pause",
    "a food delivery driver at the 47th floor, elevator broken, he is halfway up the emergency stairwell, still holding the bag perfectly level",
    "a getaway driver idling outside a bank, everything going wrong, police three blocks away, watching a street musician through the windshield",
    "a free solo climber 2000 feet up with no ropes, one move from summit, just received devastating news on her phone, now frozen against the rock",
    "a paparazzo who has just captured the career-ending photo he stalked for years, looking at it on the camera screen, not sure he will publish it",
    "a surgeon mid-operation realizing the patient is the biological father he has never met, checking the chart again, twenty minutes left in the procedure",

    # ── Cross-cultural & historical remix ───
    "a geisha in full regalia sitting in a modern subway car among salarypeople, reading a thick economics textbook, completely unbothered",
    "a samurai using a broken sword to stir convenience store ramen, waiting for it to cool, helmet off beside him on the counter, 2am",
    "an Edo-period merchant discovering a smartphone washed up on the beach, poking it with a stick, crowd of onlookers keeping safe distance",
    "a feudal battle scene but one warrior has clearly googled how to survive a sword fight on his phone between clashes, army waiting, general confused",
    "a 1920s jazz musician transported to a Tokyo underground venue, immediately invited to sit in with the band, both sides figuring it out",
    "ancient Roman soldier working a convenience store night shift, toga tucked into the apron, adapting surprisingly well, helping a customer find the ATM",
    "a medieval plague doctor being onboarded at a modern hospital, HR rep trying to explain the computer system, beak mask still on",
    "a traditional oiran hosting a podcast, ornate kanzashi in her hair, ring light, laptop, speaking earnestly about historical labor conditions",
    "a Viking navigator using GPS for the first time, deeply offended by it but taking notes anyway, ship still at sea, crew watching anxiously",
    "a Victorian noblewoman stepping off a portal into modern Tokyo, immediately being handed a Suica card by a bewildered station attendant",

    # ── Pure atmosphere & texture ────────────
    "the inside of a taiko drum maker's workshop, sawdust-heavy air, enormous skins drying on frames, close detail on the craftsman's hands, decades of callus",
    "ramen broth being made at 5am, surface shimmering with fat, the chef's face reflected in the broth surface, 30 years of muscle memory visible in his posture",
    "a soroban abacus competition, rows of children moving beads at blurred speed, intense concentration, the particular competitive silence of the room",
    "interior of a whisky barrel aging warehouse, rows vanishing into darkness, amber glow where light finds the barrels, a master distiller tasting from a pipette",
    "a vinyl record pressing plant, a record just off the press still warm, worker holding it to the light checking for defects, decades of routine in the gesture",
    "a cicada shell, empty, clinging to old bark, late afternoon light making it amber-translucent, everything else soft and unfocused",
    "an analog nuclear plant control room, 1980s dials and switches, night shift, one operator, the particular silence of total responsibility",
    "inside a washi paper maker's workshop, fibers drifting in the water bath, morning light through shoji screens, the texture of the water surface",
    "a calligrapher at the exact moment the brush leaves the paper on the final stroke of a character meaning endurance, ink still wet, hands steady",
    "a typesetter's workshop at the moment letterpress printing became obsolete, mid-print run, the owner's hand still on the press, the newspaper is about desktop publishing",

    # ── Wild cards ───────────────────────────
    "a deep sea submersible pilot, tiny viewport, something enormous just moved through the exterior floodlight beam, she is leaning forward in her seat",
    "a storm chaser parked directly in the path of an F4 tornado eating a gas station burrito, filming calmly, this is Tuesday for him",
    "a competitive eater between heats, sitting alone with sixty empty bowls, somewhere between victorious and deeply questioning his choices",
    "a forensic accountant who just found proof of massive fraud, alone in a windowless office at midnight, realizing how much danger she is now in",
    "a child prodigy chess grandmaster, age 9, sitting across from a 70-year-old former world champion, both completely in their element, silent tournament hall",
    "a monk who has taken a vow of silence accidentally winning a pub quiz, desperately trying to communicate his discomfort with the prize ceremony around him",
    "a submarine cook preparing a gourmet meal in a space the size of a closet, 300 meters underwater, the sub tilting 15 degrees, still plating with tweezers",
    "a train conductor on the last run of a decommissioned line, every station dark and empty, she keeps making the full announcements anyway, alone in the cab",
    "a magician who actually has real magic performing at a low-budget children's birthday party, bored, hiding what he can truly do, doing card tricks instead",
    "a lighthouse keeper who has maintained the light for 40 years, now fully automated, she comes anyway every night just in case, alone on the gallery",
]

NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, deformed, bad anatomy, extra limbs, "
    "watermark, text, signature, oversaturated, generic, stock photo"
)

print("Loading VAE...")
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.bfloat16
)

print("Loading SDXL pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL_ID,
    vae=vae,
    torch_dtype=torch.bfloat16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

print("Loading LoRA weights...")
pipe.load_lora_weights(LORA_MODEL_ID)
pipe.set_progress_bar_config(disable=True)

print(f"\nGenerating {len(PROMPTS)} images → {OUTPUT_DIR}/\n")

saved_captions = {}

for i, prompt in enumerate(PROMPTS, start=1):
    filename = f"{i:03d}.png"
    out_path = OUTPUT_DIR / filename

    print(f"[{i:>3}/{len(PROMPTS)}] {prompt[:80]}...")

    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=30,
        guidance_scale=7.5,
        cross_attention_kwargs={"scale": 0.85},
        generator=torch.Generator("cuda").manual_seed(42 + i),
    ).images[0]

    image.save(out_path)
    saved_captions[filename] = prompt

captions_path = OUTPUT_DIR / "captions.json"
with open(captions_path, "w", encoding="utf-8") as f:
    json.dump(saved_captions, f, indent=2, ensure_ascii=False)

array_path = OUTPUT_DIR / "captions_array.json"
with open(array_path, "w", encoding="utf-8") as f:
    json.dump(list(saved_captions.values()), f, indent=2, ensure_ascii=False)

print(f"\n✓ {len(PROMPTS)} images saved to {OUTPUT_DIR}/")
print(f"✓ captions.json        → filename → prompt mapping")
print(f"✓ captions_array.json  → plain 100-element array")