import json
import uuid
from datetime import datetime

import pytz


def dump_report_data(sid, user_id, ts, ble_device_id, report_data):
    """
        Dump generated report to db
         report_data['commonTime'], report_data['sleepTime'],
    """
    from app import engine
    now = datetime.now(pytz.utc)
    stress_score, step_count, recovery_level, processed_hr, activity_level, plot_time_data, sleep_quality, \
    windowed_nn_interval = \
        dict(zip(report_data['commonTime'], report_data['stressScores'])), \
        dict(zip(report_data['commonTime'], report_data['stepCount'])), \
        dict(zip(report_data['commonTime'], report_data['recoveryLabel'])), \
        dict(zip(report_data['commonTime'], report_data['processedHr'])), \
        dict(zip(report_data['commonTime'], report_data['activityLevel'])), \
        report_data['plotTimeData'], \
        dict(zip(report_data['sleepTime'], report_data['sleepQuality'])), report_data['nn_interval']
    report = engine.execute(
        f"select id from report_reportdata where sid_id='{sid}' and user_id='{user_id}'"
    ).fetchone()
    if not report:
        report = engine.execute(
            f"INSERT INTO report_reportdata (id, sid_id, ble_device_id, user_id,"
            f" ts, created, modified, stress_score, step_count, recovery_level, "
            f"processed_hr, plot_time_data, sleep_quality, activity_level, windowed_nn_interval) "
            f"VALUES('{str(uuid.uuid4())}', '{sid}', '{ble_device_id}',  "
            f"'{user_id}', '{ts}', '{now}', '{now}', '{json.dumps(stress_score)}', '{json.dumps(step_count)}', "
            f"'{json.dumps(recovery_level)}', '{json.dumps(processed_hr)}',"
            f" '{json.dumps(plot_time_data)}', "
            f"'{json.dumps(sleep_quality)}', '{json.dumps(activity_level)}', '{json.dumps(windowed_nn_interval)}'"
            f") RETURNING *"
        ).fetchone()
        return report
    report = engine.execute(
        f"update report_reportdata set ts='{ts}', stress_score='{json.dumps(stress_score)}',"
        f" step_count='{json.dumps(step_count)}', recovery_level='{json.dumps(recovery_level)}',"
        f" processed_hr='{json.dumps(processed_hr)}',"
        f"plot_time_data='{json.dumps(plot_time_data)}',"
        f"sleep_quality='{json.dumps(sleep_quality)}',"
        f"activity_level='{json.dumps(activity_level)}',"
        f"windowed_nn_interval='{json.dumps(windowed_nn_interval)}',"
        f" modified='{now}' where sid_id='{sid}' and user_id='{user_id}'"
    )
    return report


def generate_report(sid, user_id):
    """
        Method to generate report using movesense algo and ble data
    """
    from app.post_to_slack import post_report_error_logs_on_slack
    import traceback
    try:
        from app import engine
        from app.utils import time_from_timestamp, datetime_converter
        from .movesense_report import execute_algorithm
        print("Generating Report for sid: ", sid)
        session = engine.execute(
            f"select sid, created_at, timezone"
            f" from ble_device_blesession where sid='{sid}' and user_id='{user_id}'"
        ).fetchone()
        session_data = engine.execute(
            f"select id, rr, hr, ecg, step_count, step_count, activity_level, sleep_pos, sleep_moment, ts, ble_device_id"
            f" from ble_device_bledata where sid_id='{sid}' and user_id='{user_id}'"
        ).fetchone()
        if not session_data or not session:
            print("No session data found")
            return False
        """
            "rds": [tick, rr, hr, ecg],
            "ads": [tick, step_count, activity_level],
            "sds": [tick, sleep_pos]
        """
        input_data_frame = {
            "captured_data": {
                "hr": {
                    "ticks": list(session_data['rr'].keys()),
                    "RR in ms": list(session_data['rr'].values()),
                    "HR in BPM": list(session_data['hr'].values()),
                    "ECG QR Amplitude": list(session_data['ecg'].values())
                },
                "slp": {
                    "ticks": list(session_data['sleep_pos'].keys()),
                    "sleep pos": list(session_data['sleep_pos'].values()),
                    "sleep moment": list(session_data['sleep_moment'].values())
                },
                "act": {
                    "ticks": list(session_data['step_count'].keys()),
                    "step count": list(session_data['step_count'].values()),
                    "activity level": list(session_data['activity_level'].values())
                }
            },
            "timezone": session['timezone'],
            "Start end time": time_from_timestamp(session['created_at']),
            "Start_date_time": datetime_converter(session['created_at'])
        }
        output_data_frame = execute_algorithm(input_data_frame)
        dump_report_data(sid, user_id, session_data['ts'], session_data['ble_device_id'], output_data_frame)
        return True
    except Exception as e:
        print(e)
        post_report_error_logs_on_slack(sid, user_id, traceback.format_exc())
        return False
    except:
        return False
