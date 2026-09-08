"""A deterministic generator for the domain corpus.

Every record shipped in ``examples/`` comes from here. Nothing was scraped,
copied, or taken from an operator's maintenance system, and each record says so
in its own ``source`` and ``license`` fields.

**Why synthesise.** A corpus is the one part of a machine-learning repository
that cannot be hand-waved. Real maintenance records are somebody's commercial
data; publishing them is a licensing question with a wrong answer, and scrubbing
them well enough to publish is a larger and less certain job than generating
them. A generated corpus is also strictly better for what this repository has to
demonstrate: **the duplicates and the contamination are planted at known rates**,
so the deduplicator and the contamination gate can be asserted to find a known
answer rather than merely to produce one.

**What is modelled.** Aircraft maintenance work orders, labelled with their
ATA 100 chapter. The domain is real, its vocabulary is genuinely unlike general
English — ``strut``, ``bleed``, ``pitot``, ``actuator``, part numbers, pressures
in psi — and that difference is exactly what makes a domain tokenizer beat a
general one. Chapter assignment is a real, tedious, high-volume task that is
worth automating and that a small model does well.

**What is not claimed.** The phrasing here is a plausible imitation of
maintenance English, generated from a grammar. It is not a sample of the real
distribution, and no result in this repository should be read as a claim about
one. What the corpus supports is a claim about the *pipeline*: given data of
this shape, these are the numbers, and here is the evidence that they were
measured on data the model had not seen.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from random import Random

from dslm.corpus.record import Corpus, Record, build_corpus

#: The licence every generated record carries. CC0 because a grammar's output is
#: not meaningfully authored, and a portfolio corpus with an ambiguous licence is
#: a corpus nobody can reuse.
LICENSE = "CC0-1.0"

#: Written into every record and into the corpus header, so that a file taken
#: out of context still says what it is.
SOURCE = "synthetic:dslm-mro-grammar-v1"


@dataclass(frozen=True, slots=True)
class Chapter:
    """One ATA chapter and the vocabulary that belongs to it."""

    number: int
    name: str
    systems: tuple[str, ...]
    components: tuple[str, ...]
    symptoms: tuple[str, ...]
    actions: tuple[str, ...]


#: Twelve chapters. Enough that a majority-class baseline is weak (8.3% if the
#: corpus were balanced), few enough that every class has thousands of examples
#: at a corpus size that trains in seconds.
CHAPTERS: tuple[Chapter, ...] = (
    Chapter(
        21,
        "air conditioning",
        ("pack", "cabin air", "recirculation", "temperature control"),
        (
            "pack flow control valve",
            "mix manifold",
            "trim air valve",
            "recirculation fan",
            "cabin temperature sensor",
            "ram air inlet actuator",
        ),
        (
            "overheat indication during climb",
            "cabin temperature drifting warm",
            "flow control valve stuck open",
            "fan bearing noise on start",
            "duct overheat light illuminated",
        ),
        (
            "replaced the valve and performed an operational check",
            "cleaned the sensor and reran the temperature control test",
            "reset the controller and monitored for two sectors",
            "replaced the fan assembly and checked for abnormal noise",
        ),
    ),
    Chapter(
        24,
        "electrical power",
        ("generation", "dc power", "ac power", "battery"),
        (
            "integrated drive generator",
            "transformer rectifier unit",
            "battery charger",
            "bus tie contactor",
            "static inverter",
            "ground power receptacle",
        ),
        (
            "generator dropped offline in cruise",
            "battery discharge indication on the ground",
            "transformer rectifier over-temperature",
            "contactor failed to close on transfer",
            "intermittent bus fault recorded",
        ),
        (
            "replaced the unit and carried out a load check",
            "cleaned and re-torqued the terminal connections",
            "performed a capacity test and returned the battery to service",
            "swapped the contactor and confirmed correct transfer",
        ),
    ),
    Chapter(
        27,
        "flight controls",
        ("aileron", "elevator", "rudder", "flaps", "spoilers"),
        (
            "aileron servo actuator",
            "flap track carriage",
            "spoiler actuator",
            "rudder travel limiter",
            "elevator feel unit",
            "slat drive motor",
        ),
        (
            "asymmetry detected during flap extension",
            "stiff control input reported by the crew",
            "actuator seepage found at the rod end",
            "surface out of rig by two degrees",
            "position feedback disagreement",
        ),
        (
            "rigged the surface and recorded the rigging values",
            "replaced the actuator and completed a functional test",
            "lubricated the carriage and reran the extension cycle",
            "adjusted the feedback sensor and verified agreement",
        ),
    ),
    Chapter(
        28,
        "fuel",
        ("storage", "distribution", "indicating", "refuel"),
        (
            "boost pump",
            "crossfeed valve",
            "fuel quantity probe",
            "refuel coupling",
            "surge tank float switch",
            "fuel scavenge ejector",
        ),
        (
            "low pressure light on the left tank",
            "quantity indication drifting in cruise",
            "seepage at the coupling after refuel",
            "crossfeed valve slow to respond",
            "water found in the sump drain sample",
        ),
        (
            "replaced the pump and performed a leak check",
            "recalibrated the probe and reran the quantity check",
            "replaced the seal and confirmed no seepage after pressure refuel",
            "drained the sump and sampled until clear",
        ),
    ),
    Chapter(
        29,
        "hydraulic power",
        ("green system", "yellow system", "blue system", "reservoir"),
        (
            "engine driven pump",
            "electric hydraulic pump",
            "accumulator",
            "reservoir pressurisation valve",
            "priority valve",
            "case drain filter",
        ),
        (
            "system pressure fluctuating in the cruise",
            "reservoir quantity low at gate arrival",
            "pump overheat indication",
            "filter clog indicator popped",
            "accumulator pre-charge below limits",
        ),
        (
            "replaced the pump and bled the system",
            "serviced the reservoir and checked for external leakage",
            "replaced the filter element and reset the indicator",
            "recharged the accumulator to the specified pressure",
        ),
    ),
    Chapter(
        30,
        "ice and rain protection",
        ("wing anti-ice", "engine anti-ice", "probe heat", "windshield heat"),
        (
            "wing anti-ice valve",
            "probe heat computer",
            "windshield heat controller",
            "pitot heat element",
            "drain mast heater",
            "rain repellent nozzle",
        ),
        (
            "anti-ice valve failed to open on selection",
            "probe heat fault on the first officer side",
            "windshield delamination at the lower corner",
            "heater element open circuit",
            "drain mast frozen after a cold soak",
        ),
        (
            "replaced the valve and performed an operational check",
            "replaced the computer and confirmed heat on all probes",
            "replaced the element and measured the resistance",
            "cleared the mast and verified drainage",
        ),
    ),
    Chapter(
        32,
        "landing gear",
        ("main gear", "nose gear", "brakes", "steering", "extension and retraction"),
        (
            "shock strut",
            "brake assembly",
            "tyre",
            "uplock actuator",
            "steering servo valve",
            "proximity sensor target",
        ),
        (
            "shock strut pressure low at the gate",
            "brake wear pin below limits",
            "tyre pressure below the daily minimum",
            "gear disagree indication after retraction",
            "shimmy reported during the landing roll",
        ),
        (
            "serviced the strut with nitrogen to the extension chart",
            "replaced the brake assembly and performed a taxi check",
            "replaced the tyre and torqued the axle nut to specification",
            "rigged the proximity target and confirmed correct indication",
        ),
    ),
    Chapter(
        33,
        "lights",
        ("flight compartment", "passenger compartment", "exterior", "emergency"),
        (
            "landing light assembly",
            "navigation light",
            "strobe power supply",
            "emergency light battery pack",
            "dome light dimmer",
            "logo light",
        ),
        (
            "landing light inoperative on the left wing",
            "strobe flashing irregularly",
            "emergency light pack failing its discharge test",
            "dimmer not controlling brightness",
            "logo light lens cracked",
        ),
        (
            "replaced the assembly and confirmed correct illumination",
            "replaced the power supply and checked the flash rate",
            "replaced the battery pack and completed the discharge test",
            "replaced the lens and inspected the seal",
        ),
    ),
    Chapter(
        34,
        "navigation",
        ("air data", "inertial", "radio navigation", "surveillance"),
        (
            "air data module",
            "inertial reference unit",
            "radio altimeter transceiver",
            "transponder",
            "weather radar transceiver",
            "pitot probe",
        ),
        (
            "airspeed disagreement during the takeoff roll",
            "inertial unit failed to align",
            "radio altimeter dropping out below two hundred feet",
            "transponder no reply reported by air traffic control",
            "radar returns weak on the left side",
        ),
        (
            "replaced the module and performed a pitot static leak check",
            "replaced the unit and completed a full alignment",
            "replaced the transceiver and carried out a ramp test",
            "cleaned the connector and confirmed correct replies",
        ),
    ),
    Chapter(
        36,
        "pneumatic",
        ("bleed air", "distribution", "indicating"),
        (
            "bleed air valve",
            "precooler",
            "high pressure valve",
            "fan air valve",
            "overheat detection loop",
            "bleed duct coupling",
        ),
        (
            "bleed trip off during climb",
            "duct leak detected on the loop",
            "precooler outlet temperature high",
            "valve slow to modulate",
            "coupling seal degraded",
        ),
        (
            "replaced the valve and performed a bleed leak check",
            "replaced the coupling seal and pressure tested the duct",
            "cleaned the precooler matrix and rechecked the outlet temperature",
            "replaced the detection loop section and reran the test",
        ),
    ),
    Chapter(
        49,
        "auxiliary power",
        ("apu engine", "apu starting", "apu fuel", "apu indicating"),
        (
            "auxiliary power unit starter",
            "apu fuel control unit",
            "apu inlet door actuator",
            "apu generator",
            "apu oil cooler",
            "apu exhaust muffler",
        ),
        (
            "auxiliary power unit hung start on the ground",
            "inlet door failed to open on selection",
            "oil quantity low at the daily check",
            "high exhaust gas temperature during start",
            "generator not available after start",
        ),
        (
            "replaced the starter and completed a start check",
            "rigged the inlet door and confirmed full travel",
            "serviced the oil and inspected for leakage",
            "replaced the fuel control unit and performed a start",
        ),
    ),
    Chapter(
        52,
        "doors",
        ("passenger doors", "cargo doors", "service doors", "door warning"),
        (
            "door proximity sensor",
            "cargo door actuator",
            "slide girt bar fitting",
            "door seal",
            "door latch mechanism",
            "escape slide bottle",
        ),
        (
            "door not closed indication with the door secured",
            "cargo door slow to close on the ground",
            "seal degraded along the lower edge",
            "latch mechanism stiff to operate",
            "slide bottle pressure below the chart",
        ),
        (
            "rigged the sensor and confirmed the indication cleared",
            "replaced the actuator and performed an operational check",
            "replaced the seal and carried out a pressurisation check",
            "lubricated the mechanism and reran the operation",
        ),
    ),
)

#: Vocabulary that belongs to no chapter in particular.
#:
#: Without this the task is trivial and the repository is dishonest. The first
#: version of this grammar gave every chapter its own disjoint vocabulary, so a
#: record could be classified by spotting one keyword: **both models scored
#: 100.00%**, the comparison between them was vacuous, and the reported accuracy
#: measured the grammar rather than the model.
#:
#: Real maintenance text does not work that way. "valve", "actuator", "sensor",
#: "leak" and "overheat" appear across most ATA chapters, and a write-up made
#: entirely of those words is genuinely ambiguous — a human chapters it from
#: context, or gets it wrong. Drawing a slot from this pool instead of the
#: chapter's own makes the task require accumulating weak evidence, which is
#: what the task actually is.
GENERIC_COMPONENTS: tuple[str, ...] = (
    "control valve",
    "position sensor",
    "wiring loom",
    "connector block",
    "actuator",
    "pressure switch",
    "relief valve",
    "filter element",
    "temperature probe",
    "mounting bracket",
    "electrical harness",
    "seal",
)
GENERIC_SYMPTOMS: tuple[str, ...] = (
    "intermittent fault reported",
    "component leaking",
    "overheat indication",
    "no response to selection",
    "corrosion found during inspection",
    "chafing damage to the loom",
    "reading out of tolerance",
    "warning message displayed",
    "unit failed the functional test",
    "abnormal noise reported",
)
GENERIC_ACTIONS: tuple[str, ...] = (
    "replaced the component and performed an operational check",
    "cleaned the connector and reran the test",
    "re-torqued the fitting and confirmed no leakage",
    "replaced the seal and carried out a functional check",
    "adjusted the setting and verified within tolerance",
    "inspected and returned to service with no fault found",
)
GENERIC_SYSTEMS: tuple[str, ...] = ("indicating", "control", "distribution", "monitoring")


#: Aircraft types the corpus draws from. Carried in ``meta`` and used as the
#: grouping key for a split, so a type never straddles train and test.
AIRCRAFT: tuple[str, ...] = ("A320", "A321", "B737", "B738", "E190", "A220")

#: Where the work was raised. Bounded, and part of the record's metadata.
STATIONS: tuple[str, ...] = ("LHR", "AMS", "CDG", "MAD", "DUB", "CPH", "LIS")

#: Sentence frames.
#:
#: Several, and every one of them varying at least five slots. The first draft
#: had frames like ``"{system}: {symptom}. {action}."``, which for one chapter
#: can produce only 4 x 5 x 4 = 80 distinct strings — so at 500 records per
#: chapter the grammar collided with itself constantly and a quarter of the
#: corpus was duplicated before a single duplicate had been *planted*. That
#: made the deduplicator's measurement meaningless: it could not be said to
#: have found the planted duplicates rather than the accidental ones.
FRAMES: tuple[str, ...] = (
    "{aircraft} {station}: {system} — {symptom}. inspected the {component} and {action}.",
    "crew reported {symptom} on the {aircraft} at {station}. {component} {partno}; {action}.",
    (
        "{symptom} during the {phase} check on {aircraft}. "
        "isolated the {component} {partno}; {action}."
    ),
    "{component} {partno} ({system}) on {aircraft}: {symptom}. {action} per {manual}.",
    "found {symptom} at {station} during the {phase} check. {component}: {action} per {manual}.",
    "deferred item on {aircraft} for {symptom} ({system}). {component} {partno}: {action}.",
    "{phase} inspection at {station}: the {component} showed {symptom}. {action} per {manual}.",
    "{aircraft} {system} at {station} — {symptom}. {component} {partno}: {action}.",
)

PHASES: tuple[str, ...] = ("transit", "daily", "weekly", "a-check", "line", "ramp")


@dataclass(frozen=True, slots=True)
class Plan:
    """What to generate, and what to plant in it.

    The planted rates are the point. A deduplicator tested against a corpus with
    an unknown number of duplicates can only be observed to remove *some*; one
    tested against a corpus with 4% planted duplicates can be asserted to find
    them, and to find no more than that.
    """

    name: str
    records: int = 6000
    seed: int = 20260908
    #: Fraction of records that are exact duplicates of an earlier record.
    duplicate_rate: float = 0.03
    #: Fraction that are *near* duplicates: the same work order with a station,
    #: an aircraft or a part number changed. The case a hash-based deduplicator
    #: misses entirely and the reason MinHash is here.
    near_duplicate_rate: float = 0.04
    #: Probability that any one slot is filled from the chapter-neutral pool
    #: instead of the chapter's own vocabulary. This is what stops the task from
    #: being keyword lookup; see GENERIC_COMPONENTS for the measurement that
    #: made it necessary.
    generic_rate: float = 0.45
    #: Fraction of records whose label is wrong, as real maintenance data's is:
    #: a technician picks the wrong chapter, and nothing downstream corrects it.
    #: This puts a genuine ceiling on achievable accuracy, which is the honest
    #: setting for a repository that reports one.
    label_noise: float = 0.04
    metadata: dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        """One line, for a report."""
        return (
            f"{self.name}: {self.records} record(s), seed {self.seed}, "
            f"{self.duplicate_rate:.0%} exact and {self.near_duplicate_rate:.0%} near "
            f"duplicates, {self.generic_rate:.0%} chapter-neutral vocabulary, "
            f"{self.label_noise:.0%} label noise"
        )


#: The plans that ship. `main` is the corpus everything else is built from;
#: `holdout` is generated with a different seed and is used to show that a model
#: trained on `main` still works on data it has never seen from the same
#: process — which is a weaker claim than a real distribution shift, and is
#: labelled as such wherever it is reported.
PLANS: dict[str, Plan] = {
    "main": Plan(
        name="main",
        records=6000,
        seed=20260908,
        metadata={"shows": "the corpus everything else is built from"},
    ),
    "holdout": Plan(
        name="holdout",
        records=1500,
        seed=20260909,
        duplicate_rate=0.0,
        near_duplicate_rate=0.0,
        metadata={"shows": "unseen records from the same generator, for a final check"},
    ),
    "clean": Plan(
        name="clean",
        records=2000,
        seed=20260910,
        duplicate_rate=0.0,
        near_duplicate_rate=0.0,
        metadata={"shows": "no planted duplicates; the deduplicator's negative control"},
    ),
}


def _part_number(rng: Random) -> str:
    return f"p/n {rng.randint(100, 999)}-{rng.randint(1000, 9999)}"


def _manual(rng: Random, chapter: Chapter) -> str:
    return f"amm {chapter.number}-{rng.randint(10, 89)}-{rng.randint(1, 9):02d}"


def _pick(rng: Random, specific: tuple[str, ...], generic: tuple[str, ...], rate: float) -> str:
    """Draw from the chapter's vocabulary, or from the chapter-neutral pool."""
    return rng.choice(generic if rng.random() < rate else specific)


def _sentence(rng: Random, chapter: Chapter, generic_rate: float) -> str:
    """One work order description, from the grammar."""
    frame = rng.choice(FRAMES)
    return frame.format(
        system=_pick(rng, chapter.systems, GENERIC_SYSTEMS, generic_rate),
        component=_pick(rng, chapter.components, GENERIC_COMPONENTS, generic_rate),
        symptom=_pick(rng, chapter.symptoms, GENERIC_SYMPTOMS, generic_rate),
        action=_pick(rng, chapter.actions, GENERIC_ACTIONS, generic_rate),
        aircraft=rng.choice(AIRCRAFT),
        station=rng.choice(STATIONS),
        phase=rng.choice(PHASES),
        partno=_part_number(rng),
        manual=_manual(rng, chapter),
    )


def _vary(rng: Random, text: str) -> str:
    """Return a near-duplicate of *text*: the same work order, one detail changed.

    This is the case that separates a real deduplicator from a hash set. Two
    records differing only in the station code are, for training purposes, the
    same record — and an exact-match check sees two different strings.
    """
    for station in STATIONS:
        if station in text:
            return text.replace(station, rng.choice([s for s in STATIONS if s != station]), 1)
    for aircraft in AIRCRAFT:
        if aircraft in text:
            return text.replace(aircraft, rng.choice([a for a in AIRCRAFT if a != aircraft]), 1)
    # No station or aircraft to swap: change the part number, which is the
    # detail most often different between two otherwise identical write-ups.
    words = text.split()
    for index, word in enumerate(words):
        if word.count("-") == 1 and word[0].isdigit():
            words[index] = f"{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"
            return " ".join(words)
    return text + " (repeat finding)"


def generate(plan: Plan) -> Corpus:
    """Generate a corpus from *plan*, deterministically.

    The same plan gives the same records on every machine, forever. That is what
    makes a committed corpus checkable rather than merely present: ``dslm check``
    regenerates it and compares digests.

    ``random.Random`` rather than NumPy's generator, deliberately. Everything
    here is a choice among strings — an integer operation — so the stdlib
    Mersenne Twister is exactly reproducible across platforms and versions, with
    none of the one-ULP libm divergence that the *floating point* path in this
    project has to account for. See ADR-004.
    """
    # nosec B311 - a corpus generator has to be *reproducible*, which is the
    # opposite of what a cryptographic generator provides. See ADR-004.
    rng = Random(plan.seed)  # noqa: S311  # nosec B311
    records: list[Record] = []
    texts: list[str] = []

    exact = int(plan.records * plan.duplicate_rate)
    near = int(plan.records * plan.near_duplicate_rate)
    fresh = plan.records - exact - near
    if fresh <= 0:
        raise ValueError("the planted duplicate rates leave no fresh records to copy from")

    numbers = [chapter.number for chapter in CHAPTERS]
    for index in range(fresh):
        chapter = CHAPTERS[index % len(CHAPTERS)]
        text = _sentence(rng, chapter, plan.generic_rate)
        texts.append(text)
        # Label noise. Applied *after* the text is generated, so the text is a
        # correct description of the true chapter and only the recorded label is
        # wrong — which is how a mis-chaptered work order actually looks, and
        # what puts a real ceiling under any achievable accuracy.
        label = chapter.number
        mislabelled = rng.random() < plan.label_noise
        if mislabelled:
            label = rng.choice([number for number in numbers if number != chapter.number])
        records.append(
            _record(
                plan,
                index,
                text,
                label,
                rng,
                kind="fresh",
                truth=chapter.number if mislabelled else None,
            )
        )

    # Duplicates are appended after the fresh records rather than interleaved.
    # Order does not matter — every consumer either shuffles or works over the
    # set — and generating them in one pass keeps the seed's meaning simple.
    for offset in range(exact):
        source_index = rng.randrange(len(texts))
        index = fresh + offset
        records.append(
            _record(
                plan, index, texts[source_index], records[source_index].label, rng, kind="exact"
            )
        )

    for offset in range(near):
        source_index = rng.randrange(len(texts))
        index = fresh + exact + offset
        records.append(
            _record(
                plan,
                index,
                _vary(rng, texts[source_index]),
                records[source_index].label,
                rng,
                kind="near",
            )
        )

    return build_corpus(
        records,
        source=f"<generated:{plan.name}>",
        metadata={
            "generated": "true",
            "note": (
                "Synthesised maintenance records. Nothing here was captured from a real "
                "operator, and no result over it is a claim about real maintenance data."
            ),
            "plan": plan.name,
            "seed": str(plan.seed),
            "grammar": SOURCE,
            "planted_exact_duplicates": str(exact),
            "planted_near_duplicates": str(near),
            "generic_rate": f"{plan.generic_rate:g}",
            "label_noise": f"{plan.label_noise:g}",
            **plan.metadata,
        },
    )


def _record(  # noqa: PLR0913 - a record is its plan, index, text, label and kind
    plan: Plan,
    index: int,
    text: str,
    label: int,
    rng: Random,
    *,
    kind: str,
    truth: int | None = None,
) -> Record:
    meta = {
        "aircraft": rng.choice(AIRCRAFT),
        "station": rng.choice(STATIONS),
        # Recorded so a test can assert the deduplicator found the planted
        # duplicates and the label noise is where it claims to be. Nothing in the
        # pipeline reads either key; removing them would not change a metric.
        "kind": kind,
    }
    if truth is not None:
        meta["true_label"] = str(truth)
    return Record(
        record_id=f"{plan.name}-{index:06d}",
        text=text,
        label=label,
        source=SOURCE,
        license=LICENSE,
        meta=meta,
    )


def plan_digest(plans: Sequence[Plan]) -> str:
    """A content address over the plans, for a report."""
    material = "\n".join(
        f"{plan.name}:{plan.records}:{plan.seed}:{plan.duplicate_rate}:{plan.near_duplicate_rate}"
        for plan in plans
    )
    return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
