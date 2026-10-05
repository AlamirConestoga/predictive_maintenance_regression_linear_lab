import datetime
import pandas as pd
from sqlalchemy import Engine

def generate_work_order(axis: str, alert_type: str, deviation: float, timestamp: str, engine: Engine = None) -> dict:
    """
    Creates a maintenance work order record with priority based on alert level.
    """
    priority = "HIGH" if alert_type == "ERROR" else "MEDIUM"
    description = f"Automated Predictive Maintenance Trigger: {axis} showed sustained abnormal current. " \
                  f"Deviation: {deviation:.4f} A above regression baseline."
    
    work_order = {
        "work_order_id": f"WO-{int(datetime.datetime.now().timestamp())}",
        "created_at": str(timestamp),
        "target_component": axis,
        "priority": priority,
        "alert_type": alert_type,
        "description": description,
        "status": "OPEN",
        "assigned_team": "Robot Maintenance Crew"
    }
    
    # Optional: Save directly to PostgreSQL/Neon.tech database
    if engine:
        wo_df = pd.DataFrame([work_order])
        wo_df.to_sql('work_orders', con=engine, if_exists='append', index=False)
        print(f"[WORK ORDER CREATED] Work order {work_order['work_order_id']} saved to database.")
    
    return work_order


def send_anomaly_notification(work_order: dict):
    """
    Simulates sending an email/Slack alert to the engineering team.
    """
    message = f"""
    🚨 ANOMALY NOTIFICATION DETECTED 🚨
    -----------------------------------
    Work Order ID : {work_order['work_order_id']}
    Component     : {work_order['target_component']}
    Severity      : {work_order['priority']} ({work_order['alert_type']})
    Timestamp     : {work_order['created_at']}
    Details       : {work_order['description']}
    Action Required: Perform immediate inspection prior to hardware failure.
    -----------------------------------
    """
    print(message)