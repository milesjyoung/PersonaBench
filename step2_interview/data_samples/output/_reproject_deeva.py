import json
import io
from collections import OrderedDict

path = r"C:\Users\eclip\Desktop\mom's ai project\PersonaBench\step2_interview\data_samples\output\deeva_cintron_verification.json"

with io.open(path, "r", encoding="utf-8") as f:
    data = json.load(f, object_pairs_hook=OrderedDict)

hf = data["corrected_extracted_profile"]["hidden_facts"]

# Pass 1: mutate existing facts to enforce key order and add source/reprojected_from
new_hf = []
for fact in hf:
    ordered = OrderedDict()
    ordered["fact_id"] = fact["fact_id"]
    ordered["source"] = "direct"
    ordered["source_subcategory"] = fact["source_subcategory"]
    ordered["reprojected_from"] = None
    ordered["claim"] = fact["claim"]
    ordered["ground_truth_label"] = fact["ground_truth_label"]
    ordered["behavioral_evidence_from_interview"] = fact["behavioral_evidence_from_interview"]
    ordered["duplicate_of"] = fact.get("duplicate_of", None)
    ordered["also_supported_by"] = fact.get("also_supported_by", [])
    new_hf.append(ordered)

# Pass 2: add reprojected facts from summaries
def mk(fact_id, source_subcategory, reprojected_from, claim, gt_label, evidence_qs):
    o = OrderedDict()
    o["fact_id"] = fact_id
    o["source"] = "reprojected"
    o["source_subcategory"] = source_subcategory
    o["reprojected_from"] = reprojected_from
    o["claim"] = claim
    o["ground_truth_label"] = gt_label
    o["behavioral_evidence_from_interview"] = evidence_qs
    o["duplicate_of"] = None
    o["also_supported_by"] = []
    return o

reprojected = [
    # ---- DEMOGRAPHER ----
    mk("HF-102", "preferences.work_and_career", "summaries.demographer",
       "Household gross income ~$5,500/month composed of UAW Ford pension (Harold's), her own Social Security, survivor Social Security, and investment RMDs",
       "Gross monthly income ~$5,500 from pension + SS + RMDs",
       ["Q72", "Q73", "Q74"]),
    mk("HF-103", "preferences.work_and_career", "summaries.demographer",
       "Primary pension income source is Harold's UAW Ford pension (spousal/survivor benefit)",
       "Harold's UAW Ford pension as income anchor",
       ["Q72", "Q73"]),
    mk("HF-104", "preferences.cultural_and_linguistic", "summaries.demographer",
       "Textbook stable-Midwestern-widow demographic archetype: 58-year marriage, three children (1962-1968), long-resident homeowner, post-retirement civic actor",
       "Midwestern-widow life-course archetype",
       ["Q01", "Q05", "Q08", "Q84"]),

    # ---- BEHAVIORAL ECONOMIST ----
    mk("HF-105", "preferences.home_and_lifestyle", "summaries.behavioral_economist",
       "Exhibits extreme loss aversion and strong preference for predictability",
       "Loss-averse, predictability-preferring economic type",
       ["Q15", "Q17", "Q56", "Q63"]),
    mk("HF-106", "preferences.home_and_lifestyle", "summaries.behavioral_economist",
       "Low-discount / long-horizon time preference: paid off mortgage by 1999 via decades of below-means living",
       "Long-horizon saver, delayed-gratification pattern",
       ["Q58", "Q83"]),
    mk("HF-107", "preferences.home_and_lifestyle", "summaries.behavioral_economist",
       "Zero-slack emergency-fund tolerance: any drawdown is replenished immediately by cutting discretionary categories",
       "Zero-tolerance emergency-fund replenishment rule",
       ["Q60"]),
    mk("HF-108", "preferences.cultural_and_linguistic", "summaries.behavioral_economist",
       "Identity signaling clusters around thrift, reliability, and order ('a person's word was their bond'; 'measure a board four times before cutting')",
       "Thrift/reliability/order identity signaling",
       ["Q03", "Q24"]),
    mk("HF-109", "preferences.cultural_and_linguistic", "summaries.behavioral_economist",
       "Self-describing behavioral heuristic: 'measures a board four times before cutting' (precision-before-action trait)",
       "Precision-before-action self-description",
       ["Q03"]),
    mk("HF-110", "preferences.social_and_relationships", "summaries.behavioral_economist",
       "Peer influence is selective: resists schedule-disruption norms but updates values readily when triggered by in-group authority (Sister Mary Agnes, grandson's wife Keisha)",
       "Selective peer-influence pattern (in-group-authority updating)",
       ["Q02", "Q15", "Q26"]),
    mk("HF-111", "preferences.community_and_civic", "summaries.behavioral_economist",
       "Exhibits reciprocal altruism: monthly tithe, NAACP giving, annual gift-tax-exclusion transfers to children",
       "Reciprocal-altruism giving pattern",
       ["Q27", "Q56", "Q62"]),
    mk("HF-112", "preferences.technology", "summaries.behavioral_economist",
       "Conservative but not paralyzed risk posture: adopts new technology (Zoom, QuickBooks, CGM) only after trusted-agent mediation",
       "Trusted-agent-mediated technology adoption",
       ["Q18", "Q31", "Q48", "Q65"]),

    # ---- POLITICAL SCIENTIST ----
    mk("HF-113", "preferences.community_and_civic", "summaries.political_scientist",
       "Differentiated, high-resolution institutional trust: high for local police, PBS, CDC, and mainline Protestant institutions",
       "High trust: local police/PBS/CDC/mainline Protestant",
       ["Q20", "Q42", "Q47"]),
    mk("HF-114", "preferences.community_and_civic", "summaries.political_scientist",
       "Declining trust in the Republican Party as an institution since 2010/2016",
       "Declining GOP-institutional trust",
       ["Q24", "Q25", "Q95"]),
    mk("HF-115", "preferences.community_and_civic", "summaries.political_scientist",
       "Textbook case of stable local social-capital accumulation: 62-year home tenure, 61-year church membership, 50+-year local friendships",
       "Deep local social-capital accumulation",
       ["Q07", "Q08", "Q47"]),
    mk("HF-116", "preferences.community_and_civic", "summaries.political_scientist",
       "Splits ticket on state/local races while leaning moderate Democrat on federal races",
       "Split-ticket state/local voting pattern",
       ["Q24", "Q95", "Q96"]),

    # ---- PSYCHOLOGIST ----
    mk("HF-117", "preferences.health_and_wellness", "summaries.psychologist",
       "Personality profile: high conscientiousness, moderate neuroticism, moderate introversion, with strong need for order",
       "Big-5: high C, mod N, mod I, high order-need",
       ["Q03", "Q15", "Q45", "Q109"]),
    mk("HF-118", "preferences.health_and_wellness", "summaries.psychologist",
       "Self-describes as a 'functional worrier' (episodic anxiety that does not impair function)",
       "Self-label: functional worrier",
       ["Q45"]),
    mk("HF-119", "preferences.social_and_relationships", "summaries.psychologist",
       "Stated value rank-order: family > word > order > service > faith > perspective",
       "Rank-ordered value hierarchy",
       ["Q109"]),
    mk("HF-120", "preferences.health_and_wellness", "summaries.psychologist",
       "Emotional regulation is ritualized via self-soothing infrastructure: rigid schedule, daily gardening, morning prayer, 3 p.m. tea, evening knitting",
       "Ritualized self-soothing emotional-regulation system",
       ["Q15", "Q47", "Q50"]),
    mk("HF-121", "preferences.social_and_relationships", "summaries.psychologist",
       "Securely attached relational style with dense, long-duration ties",
       "Secure attachment, dense long-duration network",
       ["Q05", "Q07"]),
    mk("HF-122", "preferences.home_and_lifestyle", "summaries.psychologist",
       "Detail-oriented, verificationist cognitive style: double-checks crosswords and ledgers",
       "Verificationist cognitive style",
       ["Q03", "Q56"]),
    mk("HF-123", "preferences.home_and_lifestyle", "summaries.psychologist",
       "Metacognitive awareness of own rigidity and explicit effort to loosen schedule 'in Harold's honor'",
       "Metacognitive rigidity awareness + loosening effort",
       ["Q15"]),
    mk("HF-124", "preferences.life_events_and_transitions", "summaries.psychologist",
       "Generativity-dominant motivational architecture: finish degree, mentor granddaughters, preserve independence, 'be remembered as someone who paid attention and showed up'",
       "Generativity-dominant legacy motivation",
       ["Q04", "Q108", "Q109"]),
    mk("HF-125", "preferences.cultural_and_linguistic", "summaries.psychologist",
       "Value-updating on race/ethnicity was triggered by grandson Brian's 2021 interracial marriage to Keisha and subsequent engagement with BLM issues",
       "Kin-triggered race/ethnicity value update",
       ["Q26", "Q27", "Q28"]),
    mk("HF-126", "preferences.health_and_wellness", "summaries.psychologist",
       "Manages subclinical depressive/anxious symptomatology through stoic, behavioral, faith-based, and relational coping rather than clinical care",
       "Non-clinical coping strategy stack",
       ["Q44", "Q45"]),
]

new_hf.extend(reprojected)

data["corrected_extracted_profile"]["hidden_facts"] = new_hf

with io.open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# Summary
direct = sum(1 for x in new_hf if x["source"] == "direct")
reproj = sum(1 for x in new_hf if x["source"] == "reprojected")
print(f"Total: {len(new_hf)} | direct: {direct} | reprojected: {reproj}")
print("First 3 reprojected:")
for f in new_hf[direct:direct+3]:
    print(f"  {f['fact_id']} [{f['reprojected_from']}] {f['claim'][:80]}")
