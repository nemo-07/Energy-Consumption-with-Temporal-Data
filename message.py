from twilio.rest import Client

account_sid = 'AC30f56f865d94f3f68803c39d7e66a5fb'
auth_token = '5dd76e2fd7b332cd238d24319795aadb'
client = Client(account_sid, auth_token)

message = client.messages.create(
  from_='+12073379400',
  body = 'Hello',
  to='+6593530714'
)

print(message.sid)