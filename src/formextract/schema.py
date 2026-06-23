"""Pydantic schema describing the structured output of a fire-drill form.

The schema does double duty:

1. It is converted to a JSON Schema and passed to the VLM as a hard
   *constrained-decoding* grammar (``format=...`` in the Ollama call), so the
   model is forced to emit well-typed JSON rather than free prose.
2. It defines the canonical field order used to split work across the form's
   two pages (see ``vlm.get_partial_model_for_page``).
"""

from __future__ import annotations

from pydantic import BaseModel


class FireDrillFields(BaseModel):
    """One extracted fire-drill / evacuation form."""

    Site_Address: str
    DDSO_Provider_Agency: str
    Date: str
    Time_Evacuation_Started: str
    Part_of_Day: str
    Total_time_to_evacuate_to_ground: str
    Total_time_for_all_to_reach_safe_area: str
    Centrally_Monitored_Fire_Alarm_Station: str
    Time_Monitoring_Station_Notified_of_Drill: str
    Time_Monitoring_Station_Reactivated: str
    Time_Monitoring_Station_Received_Alarm: str
    Weather_Conditions: str
    Method_of_Alarm_Activation: list[str]
    Evacuation_Type: str
    Type_of_Evacuation: str
    Location_of_Simulated_Fire: str
    Blocked_Exits_by_Simulated_Fire: str
    Location: str
    Name_of_Individuals_Residing_in_the_Residence: list[str]
    including_away_at_the_Time_of_the_Evacuation: list[str]
    To_Evacuate: list[str]
    At_Safe_Area: list[str]
    Evacuation_Details: list[str]
    Description_of_evacuation: str
    Problems_noted_correction_actions_taken: str
    Did_Evacuation_proceed_in_accordance_with_evac_plan: str
    Were_all_exits_escape_route_clear_of_obstructions: str
    Did_alarms_bells_horns_strobes_function_properly: str
    Did_evacuation_time_meet_location_requirement: str
    was_drill_observed: str
