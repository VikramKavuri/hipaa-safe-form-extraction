# Evaluation report (mock predictor - harness demo)

- **Documents evaluated:** 12
- **Overall exact-match (all cells):** 87.5%
- **Macro-F1 over set fields:** 0.959

## Checkbox fields
| field                                         | type   | exact   |   F1 / sim |
|-----------------------------------------------|--------|---------|------------|
| Did_evacuation_time_meet_location_requirement | scalar | 91.7%   |      0.917 |
| Evacuation_Type                               | scalar | 91.7%   |      0.917 |
| Method_of_Alarm_Activation                    | set    | 91.7%   |      0.968 |
| Type_of_Evacuation                            | scalar | 83.3%   |      0.912 |

## Scalar (text) fields
| field                                               | exact-match   |   mean sim |   n |
|-----------------------------------------------------|---------------|------------|-----|
| was_drill_observed                                  | 100.0%        |      1     |  12 |
| Problems_noted_correction_actions_taken             | 100.0%        |      1     |  12 |
| DDSO_Provider_Agency                                | 100.0%        |      1     |  12 |
| Time_Monitoring_Station_Received_Alarm              | 100.0%        |      1     |  12 |
| Total_time_for_all_to_reach_safe_area               | 100.0%        |      1     |  12 |
| Centrally_Monitored_Fire_Alarm_Station              | 100.0%        |      1     |  12 |
| Site_Address                                        | 100.0%        |      1     |  12 |
| Location                                            | 66.7%         |      0.742 |  12 |
| Did_alarms_bells_horns_strobes_function_properly    | 75.0%         |      0.75  |  12 |
| Blocked_Exits_by_Simulated_Fire                     | 75.0%         |      0.812 |  12 |
| Type_of_Evacuation                                  | 83.3%         |      0.912 |  12 |
| Did_Evacuation_proceed_in_accordance_with_evac_plan | 83.3%         |      0.833 |  12 |
| Location_of_Simulated_Fire                          | 83.3%         |      0.903 |  12 |
| Weather_Conditions                                  | 83.3%         |      0.896 |  12 |
| Total_time_to_evacuate_to_ground                    | 91.7%         |      0.917 |  12 |
| Did_evacuation_time_meet_location_requirement       | 91.7%         |      0.917 |  12 |
| Description_of_evacuation                           | 91.7%         |      0.917 |  12 |
| Time_Monitoring_Station_Reactivated                 | 91.7%         |      0.917 |  12 |
| Date                                                | 91.7%         |      0.917 |  12 |
| Evacuation_Type                                     | 91.7%         |      0.917 |  12 |
| Were_all_exits_escape_route_clear_of_obstructions   | 91.7%         |      0.917 |  12 |
| Time_Monitoring_Station_Notified_of_Drill           | 91.7%         |      0.988 |  12 |
| Time_Evacuation_Started                             | 91.7%         |      0.988 |  12 |
| Part_of_Day                                         | 91.7%         |      0.991 |  12 |

## Set / list fields
| field                                         |   precision |   recall |    F1 | exact-set   |
|-----------------------------------------------|-------------|----------|-------|-------------|
| To_Evacuate                                   |       0.949 |    0.902 | 0.925 | 50.0%       |
| At_Safe_Area                                  |       0.949 |    0.925 | 0.937 | 58.3%       |
| Name_of_Individuals_Residing_in_the_Residence |       0.975 |    0.951 | 0.963 | 75.0%       |
| Evacuation_Details                            |       1     |    0.931 | 0.964 | 83.3%       |
| Method_of_Alarm_Activation                    |       0.938 |    1     | 0.968 | 91.7%       |
| including_away_at_the_Time_of_the_Evacuation  |       1     |    1     | 1     | 100.0%      |
