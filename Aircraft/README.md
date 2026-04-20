
📋 **Project Overview**

Managing aircraft maintenance requires balancing strict safety limits with limited hangar availability. This pipeline automates the cross-referencing of Fleet Performance Logs against Manufacturer Service Intervals to ensure 100% regulatory compliance and optimized maintenance scheduling.

**Key Features**

* Data Integrity Quality Gate: Standardizes inconsistent date formats and ID strings, identifying "orphaned" components missing manufacturer rules.

* Multi-Dimensional Risk Scoring: A weighted algorithm that ranks aircraft based on Component Grade (Primary/Secondary), Condition (Damaged/OK), and Criticality (High/Medium/Low).

* Proactive Service Forecasting: Implementation of 40%, 60%, and 80% capacity thresholds to move from reactive repairs to planned hangar slot scheduling.

* Schema Resilience: Utilizes reindexing and vectorized logic to handle "null math" traps and varying data snapshots without code changes.
--- 

🛠️ **Technical Stack**

* **Language**: Python 3.x

* **Libraries:**

   * Pandas: For data manipulation, pivoting, and merging.

   * NumPy: For vectorized threshold calculations and conditional logic.

   * Regex: For string standardization and ID sanitization.


🚀 **Pipeline Stages** 

**1. Data Sanitization & Alignment**

The "Sushi Chef" approach to data: cleaning the board first. We standardize Component_ID and Assembly_ID to remove whitespace and special characters while parsing mixed international date formats.

**2. The Audit Join**

A Left Join strategy is used to map performance data to service rules. This creates an Investigation Report for components that do not have a corresponding manufacturer limit, ensuring no part is left unmonitored.

**3. Risk Prioritization (Priority 1 & 2)** 

Using a "Split-Join" pivot method, we generate a high-visibility matrix.

* Priority 1 Score: Sum of High Criticality + Damaged + Primary Grade components.

* Priority 2 Score: Sum of Medium Criticality + Secondary Grade components.

**4. Proactive Forecasting**

We filter for components that are currently "Safe" but approaching their limits.

* **Thresholds:** 0.4 (Monitor), 0.6 (Caution), and 0.8 (Critical).

* **Goal:** Provide the Planning Department with a 30-60 day outlook for hangar slot allocation.
---

📈 **Next Steps**

* **Incorporate Qualitative Data:** Integrate technician notes and passenger feedback for a holistic safety assessment.

* **Monitor Implementation:** Track changes and measure success using KPIs like "AOG (Aircraft on Ground) Hours."

* **Iterative Improvement:** Refine service thresholds and risk weighting based on emerging fleet needs and historical failure patterns.

---
📄 **Conclusion**
By transitioning from manual reports to this automated pipeline, the fleet maintenance process moves from a reactive stance to a data-driven, proactive strategy. This reduces unplanned groundings, optimizes technician labor, and flight safety through rigorous data integrity.
