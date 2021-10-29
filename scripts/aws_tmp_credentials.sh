#!/bin/bash

if [ -z "$1" ]
  then
    echo "MFA device argument not supplied"
    exit 1
  else
   # The value is either the serial number for a hardware device (such as GAHT12345678 ) or an Amazon Resource Name (ARN) for a virtual device (such as arn:aws:iam::123456789012:mfa/user).
   MFA_DEVICE_ID=$1
fi

if [ -z "$2" ]
  then
    echo "Duration argument not supplied"
  else
    DURATION=$2 # The duration, in seconds, that the credentials should remain valid
fi

if [ "$DURATION" -lt 900 ]
  then
    echo "Acceptable durations for IAM user sessions range from 900 seconds. Setting to this value..."
    DURATION=900
fi

if [ -z "$3" ]
  then
    echo "Token argument not supplied"
    exit 1
  else
   TOKEN=$3  # The value provided by the MFA device
fi

echo "Getting temporal AWS credentials for $MFA_DEVICE_ID"

unset AWS_ACCESS_KEY_ID
unset AWS_SECRET_ACCESS_KEY
unset AWS_SESSION_TOKEN

aws sts get-session-token --serial-number $MFA_DEVICE_ID --duration-seconds $DURATION --token-code $TOKEN |
    jq -r \
        --arg aki "AWS_ACCESS_KEY_ID:" \
        --arg asak "AWS_ACCESS_KEY_ID:" \
        --arg ast "AWS_SESSION_TOKEN:" \
    '.Credentials|($aki + " " + .AccessKeyId),($asak + " " + .SecretAccessKey),($ast + " " + .SessionToken)'

DURATION_IN_MINUTES=$(awk "BEGIN {print $DURATION / 60}")
echo "Credentials remain valid for $DURATION_IN_MINUTES minutes"