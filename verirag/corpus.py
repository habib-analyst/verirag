"""Synthetic paginated corpus: NovaDyne Industries knowledge base.

Each document has explicit page numbers so citations can point to exact pages.
Content is fictional but internally consistent — the eval harness tests against it.
"""
from dataclasses import dataclass, field


@dataclass
class Page:
    doc_id: str
    page_num: int
    text: str
    section: str = ""


@dataclass
class Document:
    doc_id: str
    title: str
    doc_type: str  # manual, policy, spec, procedure, faq
    date: str
    department: str
    pages: list = field(default_factory=list)


def build_corpus() -> list:
    """Build the full NovaDyne Industries knowledge base."""
    docs = []
    D = lambda doc_id, title, doc_type, date, dept, pages: docs.append(
        Document(doc_id, title, doc_type, date, dept,
                 [Page(doc_id, i + 1, t, s) for i, (t, s) in enumerate(pages)])
    )

    # --- Product manuals ---
    D("MAN-001", "NovaDyne VX-400 Industrial Pump — Operator Manual", "manual",
      "2024-03-15", "Engineering", [
        ("The NovaDyne VX-400 is a centrifugal industrial pump rated for continuous duty at flow rates up to 400 liters per minute. The pump housing is constructed from 316L stainless steel for corrosion resistance in chemical processing environments. Maximum operating pressure is 12 bar, and the rated motor power is 15 kilowatts at 2900 RPM.",
         "Specifications"),
        ("Installation requires a level concrete pad with vibration isolation mounts. The suction line must have a minimum diameter of 80 millimeters to prevent cavitation. Electrical connection is 400V three-phase. Before first startup, fill the pump casing with the process fluid through the priming port located on the top of the volute. Never run the pump dry — dry running for more than 30 seconds will damage the mechanical seal.",
         "Installation"),
        ("Routine maintenance schedule: inspect the mechanical seal every 500 operating hours and replace every 2000 hours. Check bearing temperature monthly; normal operating range is 40 to 70 degrees Celsius. Replace the wear ring when the clearance exceeds 0.5 millimeters. Use only NovaDyne-approved seal kit SK-400-VX for replacements.",
         "Maintenance"),
        ("Troubleshooting: if the pump fails to prime, check that the suction line is airtight and the foot valve is submerged. Low flow rate usually indicates a clogged strainer or worn impeller. Excessive vibration suggests misalignment, cavitation, or a damaged bearing. The fault code E-17 on the control panel indicates motor overload — reduce discharge pressure and reset.",
         "Troubleshooting"),
    ])
    D("MAN-002", "NovaDyne HT-900 Heat Exchanger — Service Guide", "manual",
      "2024-06-01", "Engineering", [
        ("The HT-900 is a shell-and-tube heat exchanger with a thermal capacity of 900 kilowatts. It uses 120 stainless steel tubes with an outer diameter of 19 millimeters. Design pressure is 16 bar on the shell side and 10 bar on the tube side. Maximum operating temperature is 220 degrees Celsius.",
         "Specifications"),
        ("Descaling procedure: isolate the unit and drain both circuits. Circulate a 5 percent citric acid solution through the tube side for 4 hours at 60 degrees Celsius. Flush with clean water until the pH of the effluent is neutral. Inspect tube sheets for pitting before returning to service. Perform descaling annually or when the approach temperature rises more than 8 degrees above baseline.",
         "Descaling"),
        ("Gasket replacement: the HT-900 uses EPDM gaskets rated for 150 degrees continuous service. For high-temperature applications above 150 degrees, order Viton gasket kit GK-900-V. Torque the channel bolts to 85 newton-meters in a star pattern. Re-torque after 24 hours of thermal cycling.",
         "Gaskets"),
    ])
    D("MAN-003", "NovaDyne SC-50 Safety Controller — Configuration Manual", "manual",
      "2024-09-10", "Engineering", [
        ("The SC-50 is a SIL-2 rated safety controller with 16 digital inputs and 8 relay outputs. It executes the safety program every 10 milliseconds. The controller supports PROFIsafe communication for integration with Siemens safety PLCs. Firmware version 3.2 or later is required for the dual-channel encoder monitoring function.",
         "Specifications"),
        ("Wiring: safety inputs must use shielded cable with the shield grounded at the controller end only. Wire the emergency stop as a dual-channel normally-closed loop. The maximum cable length for safety inputs is 100 meters. Relay outputs are rated for 6 amps at 250V AC resistive load.",
         "Wiring"),
        ("Proof testing: execute the automated proof test every 12 months using the SC-50 diagnostic software. The test exercises all inputs, outputs, and the internal watchdog. Record the proof test date in the safety logbook. A failed proof test requires immediate investigation — do not restart the process until the fault is resolved.",
         "Proof testing"),
    ])

    # --- Safety procedures ---
    D("SAF-001", "Lockout-Tagout Procedure — NovaDyne Facilities", "procedure",
      "2024-01-20", "Safety", [
        ("Lockout-tagout (LOTO) is mandatory before servicing any equipment with stored energy. The authorized employee must notify all affected workers before applying locks. Each worker applies their own personal lock — never remove another person's lock. Locks are red for electrical, blue for mechanical, and yellow for pneumatic energy sources.",
         "General"),
        ("Sequence: (1) identify all energy sources, (2) shut down equipment using normal stopping procedures, (3) isolate energy by operating disconnects and valves, (4) apply locks and tags, (5) release stored energy by bleeding lines and discharging capacitors, (6) verify isolation by attempting to start the equipment. Document each step on form LOTO-7.",
         "Sequence"),
        ("Group lockout: when multiple crafts work on the same equipment, use a group lockbox. Each worker places their personal lock on the lockbox. The primary authorized employee coordinates the sequence. Shift changes require a documented transfer of lockout responsibility — the outgoing and incoming authorized employees must both verify the isolation.",
         "Group lockout"),
    ])
    D("SAF-002", "Chemical Spill Response — Level 2 Incidents", "procedure",
      "2024-04-05", "Safety", [
        ("A Level 2 chemical spill involves 20 to 200 liters of hazardous material, or any quantity of highly toxic substances. Evacuate all personnel within a 25-meter radius immediately. Activate the spill alarm at the nearest pull station. The spill response team must arrive within 10 minutes of the alarm.",
         "Classification"),
        ("Containment: deploy absorbent booms around the spill perimeter before attempting cleanup. For acid spills, apply neutralizing agent from spill kit SK-2 before absorption. Never use combustible absorbents on oxidizer spills. Collect all contaminated material in labeled hazardous waste drums — never pour neutralized waste down drains.",
         "Containment"),
        ("Reporting: file incident report IR-12 within 24 hours. Spills exceeding 50 liters require notification to the environmental authority within 48 hours. Preserve the scene for investigation until the safety manager releases it. Review the response in the monthly safety meeting.",
         "Reporting"),
    ])

    # --- HR policies ---
    D("HR-001", "NovaDyne Employee Handbook — Leave Policy", "policy",
      "2024-02-01", "HR", [
        ("Full-time employees accrue 2.5 days of annual leave per month, totaling 30 days per year. Leave accrues from the first day of employment but cannot be taken during the 90-day probation period. Unused leave up to 10 days may be carried forward to the next calendar year; excess leave is forfeited on December 31.",
         "Annual leave"),
        ("Sick leave: 12 days per year at full pay, no medical certificate required for absences under 3 days. For absences of 3 or more consecutive days, a doctor's note must be submitted within 5 working days of return. Unused sick leave does not carry forward and is not paid out on termination.",
         "Sick leave"),
        ("Parental leave: 16 weeks at full pay for the primary caregiver, 4 weeks for the secondary caregiver, available after 12 months of service. Apply at least 8 weeks before the expected date. Parental leave can be combined with accrued annual leave for an extended absence.",
         "Parental leave"),
    ])
    D("HR-002", "Remote Work Policy v3", "policy",
      "2024-07-15", "HR", [
        ("Eligible roles may work remotely up to 3 days per week with manager approval. Core collaboration hours are 10:00 to 15:00 local time, during which remote workers must be reachable. New hires must complete 6 months on-site before remote eligibility begins.",
         "Eligibility"),
        ("Equipment: the company provides a laptop, monitor, and 500 euro home-office stipend every 3 years. Internet costs are reimbursed up to 40 euro per month with receipts. Ergonomic assessments are available on request through the facilities team.",
         "Equipment"),
    ])
    D("HR-003", "Performance Review Cycle", "policy",
      "2024-01-10", "HR", [
        ("Performance reviews run on a semi-annual cycle: H1 reviews in July, H2 reviews in January. Employees complete a self-assessment, then meet with their manager for a calibration discussion. Ratings use a 5-point scale: 1 (needs improvement) through 5 (exceptional). A rating of 2 or below triggers a performance improvement plan.",
         "Cycle"),
        ("Promotion criteria: two consecutive ratings of 4 or above, demonstrated mentorship of junior staff, and completion of the leadership fundamentals course. Promotions take effect in April or October. Salary adjustments from promotions range from 8 to 15 percent.",
         "Promotions"),
    ])

    # --- Engineering specs ---
    D("SPEC-001", "Warehouse Racking Load Specification", "spec",
      "2024-05-20", "Engineering", [
        ("Standard pallet racking bays are rated for 2500 kilograms per beam level, with a maximum of 4 beam levels per bay. The floor loading limit in the main warehouse is 5000 kilograms per square meter. Seismic bracing is required for all racking above 3 meters in zones 3 and 4.",
         "Load ratings"),
        ("Inspection: racking must be inspected annually by a certified inspector. Damaged uprights must be replaced, not repaired — orange safety pins indicate a bay is quarantined pending inspection. Report any forklift impact immediately, even if no visible damage is present.",
         "Inspection"),
    ])
    D("SPEC-002", "Cleanroom Classification Standard", "spec",
      "2024-08-12", "Engineering", [
        ("NovaDyne cleanrooms are classified to ISO 14644-1. The electronics assembly cleanroom is ISO Class 7, allowing a maximum of 352,000 particles of 0.5 micron per cubic meter. The optics cleanroom is ISO Class 5, allowing 3,520 particles of 0.5 micron per cubic meter.",
         "Classification"),
        ("Gowning procedure for ISO Class 5: don hood, coverall, boots, and sterile gloves in the gowning airlock. No cosmetics, no exposed skin. Wipe down all tools with 70 percent isopropyl alcohol before entry. Maximum occupancy is 6 persons to maintain particle counts.",
         "Gowning"),
    ])

    # --- IT ---
    D("IT-001", "VPN and Remote Access Guide", "faq",
      "2024-03-01", "IT", [
        ("Connect to the corporate VPN using the NovaDyne Secure Client, available in the software portal. Authentication requires your employee ID plus a hardware security key — SMS codes were deprecated in January 2024. The VPN gateway address is vpn.novadyne.example. Split tunneling is disabled; all traffic routes through the corporate firewall.",
         "Setup"),
        ("Troubleshooting: if the client shows error 619, your security key certificate has expired — request renewal from IT. Error 809 indicates UDP port 4500 is blocked; switch to TCP mode in client settings. Connection drops every 8 hours are expected; the client auto-reconnects within 60 seconds.",
         "Troubleshooting"),
    ])
    D("IT-002", "Data Retention Schedule", "policy",
      "2024-06-20", "IT", [
        ("Email is retained for 7 years in the compliance archive. Project files in SharePoint are retained for 10 years after project closure. Backup tapes rotate on a 90-day cycle with monthly archives kept for 3 years. Personal drives (H:) are not backed up — store work documents in SharePoint.",
         "Retention"),
        ("Deletion requests: under data protection regulations, employees may request deletion of personal data not subject to legal hold. Submit form DSR-3 to the data protection officer. Legal holds suspend all deletion — destroying documents under hold is a dismissible offense.",
         "Deletion"),
    ])

    # --- Finance ---
    D("FIN-001", "Travel Expense Policy", "policy",
      "2024-02-15", "Finance", [
        ("Domestic travel per diem is 65 euro per day; international per diem is 120 euro per day. Hotel ceiling is 150 euro domestic, 220 euro international — exceptions require VP approval. Flights over 6 hours may be booked in premium economy; business class requires C-level approval.",
         "Per diem"),
        ("Submit expenses within 30 days of travel via the expense portal. Receipts are required for all expenses over 25 euro. Mileage is reimbursed at 0.35 euro per kilometer. Advances are available for trips exceeding 2000 euro estimated cost.",
         "Submission"),
    ])
    D("FIN-002", "Purchase Approval Thresholds", "policy",
      "2024-04-01", "Finance", [
        ("Purchases under 1000 euro require manager approval only. From 1000 to 10000 euro, department head approval plus one competitive quote. From 10000 to 50000 euro, VP approval plus three quotes. Above 50000 euro, CFO approval and a formal tender process.",
         "Thresholds"),
        ("Emergency purchases up to 5000 euro may be approved retroactively within 48 hours by the department head. IT hardware always requires IT sign-off regardless of amount, to ensure security compliance.",
         "Emergency"),
    ])

    # --- Operations ---
    D("OPS-001", "Forklift Safety Rules", "procedure",
      "2024-05-11", "Operations", [
        ("Only certified operators may drive forklifts. Certification requires an 8-hour course plus a practical exam, renewed every 3 years. The speed limit inside the warehouse is 8 kilometers per hour, 5 in pedestrian zones. Always sound the horn at intersections and when reversing.",
         "Certification"),
        ("Load limits: check the data plate on every truck — typical capacity is 2500 kilograms at 500 millimeter load center. Never exceed the rated capacity. Tilt the mast back slightly when traveling with a load. Lower forks to 10 centimeters above the floor when moving.",
         "Load limits"),
        ("Pedestrian safety: pedestrians always have right of way. Maintain a 3-meter exclusion zone around operating forklifts, marked by floor tape. High-visibility vests are mandatory in the warehouse. Report all near-misses within 24 hours via form NM-4.",
         "Pedestrians"),
    ])
    D("OPS-002", "Inventory Cycle Count Procedure", "procedure",
      "2024-09-02", "Operations", [
        ("Cycle counts replace annual wall-to-wall inventory. A-class items (80 percent of value) are counted monthly, B-class quarterly, C-class semi-annually. Counts are performed by two-person teams — one counts, one records — and discrepancies over 2 percent trigger a recount.",
         "Schedule"),
        ("Blind counts: counters receive location lists without expected quantities to prevent bias. Record the actual count on form CC-9. The inventory controller investigates variances above 500 euro before posting adjustments. All adjustments require supervisor sign-off.",
         "Blind counts"),
    ])

    return docs


def get_all_pages(docs: list) -> list:
    """Flatten documents into pages."""
    pages = []
    for d in docs:
        pages.extend(d.pages)
    return pages
