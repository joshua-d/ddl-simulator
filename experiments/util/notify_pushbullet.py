from os import access
from pushbullet import PushBullet


def notify_pushbullet(title, text):
    try:
        access_token = 'o.usn8KMIrByNJ9VuRm2aybOsbnvEb4Npv'
        pb = PushBullet(access_token)
        pb.push_note(title, text)
    except:
        print('Failed to notify pushbullet')
