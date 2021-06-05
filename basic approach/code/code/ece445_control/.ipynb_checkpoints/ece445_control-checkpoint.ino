const int valvePin = 3;
const int pumpPin = 5;
int incomingByte = 0;
int lastByte = 0;
  
void setup() {
  // make the pin with the optocoupler an output
  pinMode(valvePin, OUTPUT);
  pinMode(pumpPin, OUTPUT);
  Serial.begin(9600);
  delay(1000);
  analogWrite(valvePin, 255);
  analogWrite(pumpPin, 255);
}

void loop() {
  if (Serial.available() > 0) {
    // read the incoming byte:
    if (incomingByte != 10) {
      lastByte = incomingByte;
    }
    incomingByte = Serial.read();
    Serial.print("lastByte: ");
    Serial.println(lastByte);
    // say what you got:
    if(incomingByte == '0') {
      Serial.println("Steady");
      if (lastByte == '2') {
        analogWrite(valvePin, 0);
      } else if (lastByte == '1') {
        analogWrite(valvePin, 255);
      }
      analogWrite(pumpPin, 255);
    } else if (incomingByte == '2') {
      Serial.println("Action 2");        
      analogWrite(valvePin, 255);
      analogWrite(pumpPin, 0);
    } else if (incomingByte == '1') {
      Serial.println("Action 1");
      analogWrite(valvePin, 0);
      analogWrite(pumpPin, 0);
    }
  }
  delay(100);
}
