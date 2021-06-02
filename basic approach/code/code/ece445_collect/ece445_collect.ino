/*
  * ----------------------------------------------------------------------------------------------------
  * Made for 6 chanel EMG that: 
  * - sends data for SpikeRecorder application 
  * - receive messages from SpikeRecorder application
  * - monitor all six channels at all time
  * - update VU meters for 6 ch all the time
  * - update digital outputs D7, D8, D9, D10, D11, D12  based on EMG
  * - controlls color coded LED with dimming
  * 
  * V1.0
  * Written by Jeason
  *
  * 
  * D0, D1                        - serial communication
  * 
  * D2                            - data for SHIFT registers
  * D3                            - clock for SHIFT registers
  * D4                            - latch for shift registers
  * 
  * D5                            - button that turns ON the color coded channel's LEDs
  * D6                            - The button to reserve
  *
  * D7, D8, D9, D10, D11, D12     - outputs to ESP module
  * A0, A1, A2, A3, A4, A5        - 6ch. EMG envelope inputs
  * 
  * D13                           - enable pin for color coded LEDs
  * ----------------------------------------------------------------------------------------------------
  */


#ifndef cbi
#define cbi(sfr, bit) (_SFR_BYTE(sfr) &= ~_BV(bit))
#endif
#ifndef sbi
#define sbi(sfr, bit) (_SFR_BYTE(sfr) |= _BV(bit))
#endif

#define SIZE_OF_COMMAND_BUFFER 30        //command buffer size

#define SHIFT_LATCH_PIN 4                //latch pin for shift register
#define SHIFT_CLOCK_PIN 3                //clock pin for shift register
#define SHIFT_DATA_PIN 2                 //serial data pin for shift register
#define TURN_ON_LEDS_BUTTON 5            //button on pin 5 that turns on color coded LEDs
#define COLOR_CODED_LEDS_ENABLE_PIN 13   //pin that enables color coded LEDs
#define MODE_PREPARE_SHIFT_REGISTERS 0
#define SHIFT_OUT_SHIFT_REGISTERS 1

#define COLOR_CODED_LEDS_TIME 1200000L       //120sec

#define ANTI_FLICKERING_TIME 500             //one msec is 10 so 500 is 50ms
#define SENSITIVITY_THRESHOLD 520
uint16_t antiFlickeringTimerForOutput = ANTI_FLICKERING_TIME;


byte vuMeterMode = MODE_PREPARE_SHIFT_REGISTERS;
//registers that contain LED state
byte shiftRegBytes[5];

#define MAX_LED_TIMER_FOR_PWM_DECAY 750
uint16_t ledTimerForPWMDecay = MAX_LED_TIMER_FOR_PWM_DECAY;
byte pwmThrehold = 128;
byte pwmTime = 0;
char commandBuffer[SIZE_OF_COMMAND_BUFFER];    //receiving command buffer 30

volatile  byte regularChannelsIndex;
volatile  byte roundRobinChannelIndex;
volatile  byte adcInterruptIndex;
volatile  byte lastADCIndex;

volatile  byte ISRADCFALGE;


//Set how many analogue inputs to read, starting from A0
byte numberOfChannels = 1;
byte numberOfChannelsPlusOne = numberOfChannels+1;
#define MAX_NUMBER_OF_CHANNELS 6

//Create two ping-pong buffers
int interrupt_Number = 198;// Output Compare Registers  value = (16*10^6) / (Fs*8) - 1  set to 1999 for 1000 Hz sampling, set to 3999 for 500 Hz sampling, set to 7999 for 250Hz sampling, 199 for 10000 Hz Sampling

unsigned long countdownTimerForLeds  = 0;// =  COLOR_CODED_LEDS_TIME;

volatile uint16_t samplingBuffer[MAX_NUMBER_OF_CHANNELS];//main buffer that contains real measurements
volatile uint16_t envelopeBuffer[MAX_NUMBER_OF_CHANNELS];//buffer that contains envelope of measurements


byte outputBufferReady = 0;
byte outputFrameBuffer[MAX_NUMBER_OF_CHANNELS*2];

byte vuMeterIndex;
byte ledIndex;
uint16_t movingThresholdSum;
volatile byte registerIndex = 0;

byte bitMask = 1;
 
void setup()
{
  Serial.begin(115200);      //begin Serial comm
  delay(300); //whait for init of serial
 // Serial.println("StartUp!");
 // Serial.setTimeout(2);

  //set pins to output for shift register
  pinMode(SHIFT_LATCH_PIN, OUTPUT);
  pinMode(SHIFT_CLOCK_PIN, OUTPUT);
  pinMode(SHIFT_DATA_PIN, OUTPUT);
   pinMode(TURN_ON_LEDS_BUTTON, INPUT);
  pinMode(COLOR_CODED_LEDS_ENABLE_PIN, OUTPUT);
  digitalWrite(COLOR_CODED_LEDS_ENABLE_PIN, HIGH); 

  //activate LEDS for color coding
  ledTimerForPWMDecay = MAX_LED_TIMER_FOR_PWM_DECAY;  // 750
  pwmThrehold = 128;
  countdownTimerForLeds = COLOR_CODED_LEDS_TIME;   //1200000L


  //init ESP output pins to output and to HIGH
  //since ESP is active LOW
 
  pinMode(7, OUTPUT); 
  digitalWrite(7, HIGH);  
  pinMode(8, OUTPUT); 
  digitalWrite(8, HIGH);   
  pinMode(9, OUTPUT); 
  digitalWrite(9, HIGH);  
  pinMode(10, OUTPUT); 
  digitalWrite(10, HIGH); 
  pinMode(11, OUTPUT); 
  digitalWrite(11, HIGH); 
  pinMode(12, OUTPUT); 
  digitalWrite(12, HIGH); 

    
  clearAllLeds();
  
  cli();//stop interrupts
  
  cbi(ADMUX,REFS0);  // Set ADC reference to AVCC
  cbi(ADMUX,ADLAR);// Left Adjust the result
  sbi(ADCSRA,ADEN);// Enable ADC
  sbi(ADCSRA,ADIE);// Enable ADC Interrupt

  //set ADC clock division to 16  
  sbi(ADCSRA,ADPS2);//1
  cbi(ADCSRA,ADPS1);//0
  cbi(ADCSRA,ADPS0);//0


 //set timer1 interrupt at 10kHz
  TCCR1A = 0;// set entire TCCR1A register to 0
  TCCR1B = 0;// same for TCCR1B
  TCNT1  = 0;//initialize counter value to 0;
  OCR1A = interrupt_Number;// Output Compare Registers  
  // turn on CTC mode
  TCCR1B |= (1 << WGM12);
  // Set CS11 bit for 8 prescaler
  TCCR1B |= (1 << CS11);   
  // enable timer compare interrupt
  TIMSK1 |= (1 << OCIE1A);


   numberOfChannels = 1;
   regularChannelsIndex = 0;
   roundRobinChannelIndex = numberOfChannels;
   adcInterruptIndex = 0;
   lastADCIndex = 0;
   ISRADCFALGE = 0;

   vuMeterIndex = 0;
   ledIndex = 0;
   movingThresholdSum = 0;
   registerIndex = 0;
   bitMask = 1;

  sei();                     // Enable Global Interrupts

}

//----------------------------- Main Loop ----------------------------------------------------
void loop()
{
   if(outputBufferReady == 1)
   {
     //PORTB |= 0x1;
     
     //Serial.write(outputFrameBuffer, (numberOfChannels<<1));
         
     outputBufferReady = 0;

     //------------------------------------ MAIN LOOP calculate envelope/low pass -------------------------------------- 
     //(it instantly jumps to higher value but it decay at linear rate ~ 5V/100ms)
  
      if(envelopeBuffer[0]<samplingBuffer[0])
      {
          envelopeBuffer[0]=samplingBuffer[0];
      }
      if(envelopeBuffer[1]<samplingBuffer[1])
      {
          envelopeBuffer[1]=samplingBuffer[1];
      }
      if(envelopeBuffer[2]<samplingBuffer[2])
      {
          envelopeBuffer[2]=samplingBuffer[2];
      }
      if(envelopeBuffer[3]<samplingBuffer[3])
      {
          envelopeBuffer[3]=samplingBuffer[3];
      }
      if(envelopeBuffer[4]<samplingBuffer[4])
      {
          envelopeBuffer[4]=samplingBuffer[4];
      }
      if(envelopeBuffer[5]<samplingBuffer[5])
      {
          envelopeBuffer[5]=samplingBuffer[5];
      }


     //-------------------------------- MAIN LOOP calculate what diode needs to be ON for VU meter ---------------------------------------
     if(vuMeterMode == MODE_PREPARE_SHIFT_REGISTERS)
     {
        movingThresholdSum = 10;//reset threshold
        for(ledIndex = 0;ledIndex<6;ledIndex++)
        {
          //check if voltage is higher than threshold
            if(envelopeBuffer[vuMeterIndex]>movingThresholdSum)
            {
              shiftRegBytes[registerIndex] |= bitMask;//turn ON LED
            }
            else
            {
              shiftRegBytes[registerIndex] &= ~bitMask;//Turn OFF LED
            }
            movingThresholdSum+=171;//increment voltage threshold for next LED

            bitMask = bitMask<<1; //move to next bit/LED
            if(bitMask ==0)//we got to the end of register 
            {
               //move to first bit of next register
                registerIndex++;
                bitMask = 1;
            }
        }
    
        vuMeterIndex++;
        if(vuMeterIndex == 6)
        {
          //we finish one cycle of making data for Shift registers
          //now we have to shift them out. Next time in "loop" we will
          //be in SHIFT_OUT_SHIFT_REGISTERS mode
            vuMeterIndex = 0;
            bitMask = B10000000;//prepare for bit shifting
            registerIndex = 4;//have to shift from last to first
            vuMeterMode = SHIFT_OUT_SHIFT_REGISTERS;
            digitalWrite(SHIFT_LATCH_PIN, LOW);
        }
     }
     else
     {
      //-------------------------------- MAIN LOOP shift out one bit of data --------------------------------------------------- 
        //shift out one bit of data of one register at the time, next time we will shift next one
        //until we shift all 8 for all 5 registers. Than we will change mode to MODE_PREPARE_SHIFT_REGISTERS
        //we do this to save time
        
        //shifting one bit of data to shift registers
        if(shiftRegBytes[registerIndex]&bitMask)
        {
            PORTD |=  B00000100;
        }
        else
        {
            PORTD &=  B11111011;
        }
         //pulse the clock for shift
         PORTD |=  B00001000;
         PORTD &=  B11110111;

         //move to next bit
         bitMask = bitMask>>1;

        //if we are finished with one whole register
        if(bitMask==0)
        {
          //prepare for MSB for next register
          bitMask = B10000000;

          //if we are finished with last register 
          if(registerIndex == 0)
          {
            //prepare for data preparation
              bitMask = 1;
              vuMeterMode = MODE_PREPARE_SHIFT_REGISTERS;
              digitalWrite(SHIFT_LATCH_PIN, HIGH);
          }
          else
          {
            //move to another register to shift
            registerIndex--;
          }
        }
       
     }



      //-------------------------------- MAIN LOOP    do the decay of envelope ------------------------------------------------
      //(linear rate one unit per one period of timer @10kHz ~ 5V/100ms)
      if(envelopeBuffer[0]>0)
      {
        envelopeBuffer[0]--;  
      }
      if(envelopeBuffer[1]>0)
      {
        envelopeBuffer[1]--;  
      }
      if(envelopeBuffer[2]>0)
      {
        envelopeBuffer[2]--;  
      }
      if(envelopeBuffer[3]>0)
      {
        envelopeBuffer[3]--;  
      }
      if(envelopeBuffer[4]>0)
      {
        envelopeBuffer[4]--;  
      }
      if(envelopeBuffer[5]>0)
      {
        envelopeBuffer[5]--;  
      }


      //-------------------------------- MAIN LOOP  LED enable pin ----------------------------------------------------------------
       if(countdownTimerForLeds>0)
        {
          countdownTimerForLeds--;

          //dimming the diodes at the end
          if(countdownTimerForLeds <100000)
          {
              ledTimerForPWMDecay--;
              if(ledTimerForPWMDecay==0)
              {
                
                ledTimerForPWMDecay = MAX_LED_TIMER_FOR_PWM_DECAY;
                if(pwmThrehold > 0)
                {
                    pwmThrehold--;
                }
              }
              pwmTime++;
              if(pwmTime<pwmThrehold)
              {
                PORTB |= B00100000;
              }
              else
              {
                PORTB &= B11011111;
              }
              if(pwmTime>128)
              {
                pwmTime = 0;  
              }
          }
          else
          {
            PORTB |= B00100000; //turn ON LED enable pin 13  
          }
          
        }
        else
        {
           PORTB &= B11011111; //turn ON LED enable pin 13  
        }
        //read button at pin 5
        if(digitalRead(TURN_ON_LEDS_BUTTON)==HIGH)
        {
            ledTimerForPWMDecay = MAX_LED_TIMER_FOR_PWM_DECAY;
            pwmThrehold = 128;
            countdownTimerForLeds = COLOR_CODED_LEDS_TIME;
        }
        

        //---------------------- MAIN LOOP Output pins -----------------------------------------------------------------------


        antiFlickeringTimerForOutput--;
        if(antiFlickeringTimerForOutput==0)
        {
            antiFlickeringTimerForOutput = ANTI_FLICKERING_TIME;
            if(envelopeBuffer[0]<SENSITIVITY_THRESHOLD)
            {
                PORTD |=B10000000;
            }
            else
            {
              PORTD &=B01111111;
            }


            if(envelopeBuffer[1]<SENSITIVITY_THRESHOLD)
            {
                PORTB |=B00000001;
            }
            else
            {
              PORTD &=B11111110;
            }

            if(envelopeBuffer[2]<SENSITIVITY_THRESHOLD)
            {
                PORTB |=B00000010;
            }
            else
            {
              PORTD &=B11111101;
            }

            if(envelopeBuffer[3]<SENSITIVITY_THRESHOLD)
            {
                PORTB |=B00000100;
            }
            else
            {
              PORTD &=B11111011;
            }

            if(envelopeBuffer[4]<SENSITIVITY_THRESHOLD)
            {
                PORTB |=B00001000;
            }
            else
            {
              PORTD &=B11110111;
            }

            if(envelopeBuffer[5]<SENSITIVITY_THRESHOLD)
            {
                PORTB |=B00010000;
            }
            else
            {
              PORTD &=B11101111;
            }
            
        }

    // PORTB &= ~0x1;
   }

   if(ISRADCFALGE == 1)
     {
      envelopeISRADC();
      ISRADCFALGE = 0;     
      }

}//end of loop


void envelopeISRADC()
{
      Serial.print(samplingBuffer[0]);Serial.print(",");
      Serial.print(samplingBuffer[1]);Serial.print(",");
      Serial.print(samplingBuffer[2]);Serial.print(",");
      Serial.print(samplingBuffer[3]);Serial.print(",");
      Serial.print(samplingBuffer[4]);Serial.print(",");
      Serial.print(samplingBuffer[5]);Serial.print(",");  
      Serial.println(); 
}


//------------------------- Receive serial event ----------------------------------------------
void serialEvent() 
{
 // commandMode = 1;//frag that we are receiving commands through serial
  TIMSK1 &= ~(1 << OCIE1A);//disable timer for sampling
  // read untill \n from the serial port:
  String inString = Serial.readStringUntil('\n');

  //convert string to null terminate array of chars
  inString.toCharArray(commandBuffer, SIZE_OF_COMMAND_BUFFER);
  commandBuffer[inString.length()] = 0;
  
  
  // breaks string str into a series of tokens using delimiter ";"
  // Namely split strings into commands
  char* command = strtok(commandBuffer, ";");
  while (command != 0)
  {
      // Split the command in 2 parts: name and value
      char* separator = strchr(command, ':');
      if (separator != 0)
      {
          // Actually split the string in 2: replace ':' with 0
          *separator = 0;
          --separator;
          if(*separator == 'c')//if we received command for number of channels
          {
            separator = separator+2;
            numberOfChannels = (byte)atoi(separator);//read number of channels
          }
           if(*separator == 's')//if we received command for sampling rate
          {
            //do nothing. Do not change sampling rate at this time.
            //We calculate sampling rate further below as (max Fs)/(Number of channels)
          }
          
      }
      // Find the next command in input string
      command = strtok(0, ";");
  }
  //calculate sampling rate

   regularChannelsIndex = 0;
   roundRobinChannelIndex = numberOfChannels;
   adcInterruptIndex = 0;
   lastADCIndex = 0;
  OCR1A = (interrupt_Number+1)*numberOfChannels - 1;
  TIMSK1 |= (1 << OCIE1A);//enable timer for sampling
  //commandMode = 0;
}



//------------------------- Timer interrupt ---------------------------------
ISR(TIMER1_COMPA_vect) {
 // PORTB |= 0x2;

  // Start A2D Conversions 
  //do this at the begining since ADC can work in 
  //paralel with this timer handler
  ADCSRA |=B01000000; 
  
  //convert data to frame according to protocol
  outputFrameBuffer[0]= (samplingBuffer[0]>>7)| 0x80;
  outputFrameBuffer[1]=  samplingBuffer[0] & 0x7F;
  outputFrameBuffer[2]= (samplingBuffer[1]>>7)& 0x7F;
  outputFrameBuffer[3]=  samplingBuffer[1] & 0x7F;
  outputFrameBuffer[4]= (samplingBuffer[2]>>7)& 0x7F;
  outputFrameBuffer[5]=  samplingBuffer[2] & 0x7F;
  outputFrameBuffer[6]= (samplingBuffer[3]>>7)& 0x7F;
  outputFrameBuffer[7]=  samplingBuffer[3] & 0x7F;
  outputFrameBuffer[8]= (samplingBuffer[4]>>7)& 0x7F;
  outputFrameBuffer[9]=  samplingBuffer[4] & 0x7F;
  outputFrameBuffer[10]= (samplingBuffer[5]>>7)& 0x7F;
  outputFrameBuffer[11]=  samplingBuffer[5] & 0x7F;
  //signal main loop to send frame
  outputBufferReady = 1;
  //PORTB &= ~0x2;
}

//--------------------------- ADC interrupt -----------------------------------------
//This is called when ADC conversion is complete.
//Goal is to for given frequency of sampling and given 
//number of channels every ......

ISR(ADC_vect)           
 {
     // PORTB |= 0x1;
     
       samplingBuffer[lastADCIndex] = ADCL | (ADCH << 8);// store lower and higher byte of ADC
       
       ISRADCFALGE = 1;

      if(adcInterruptIndex <numberOfChannels)
      {
        //we have to sample channels that are currently in use
        //channels that needs to be sampled every period of timer

         ADMUX =  B01000000 | regularChannelsIndex; //set ADC to A0 channel
         lastADCIndex = regularChannelsIndex;
         regularChannelsIndex++;
         if(regularChannelsIndex == numberOfChannels)
         {
            regularChannelsIndex = 0;
         }
      }
      else
      {
          //we have to sample channels that are not currently in use
          //we sample them at slower pace, one each timer period (round robin scheme)

           ADMUX =  B01000000 | roundRobinChannelIndex; //Select ADC Channel/ REFS0 set means AVcc is reference
           lastADCIndex = roundRobinChannelIndex;
           roundRobinChannelIndex++;
           if(roundRobinChannelIndex==MAX_NUMBER_OF_CHANNELS)
           {
             roundRobinChannelIndex = numberOfChannels;  
           }
      
      }    
    
      adcInterruptIndex++;
      if(adcInterruptIndex>numberOfChannels)
      {
          //if we sampled all channels that are in use
          //and one unused channel do not trigger ADC
          //but whait for next timer period and timer will trigger it
          adcInterruptIndex = 0;
          
      }
      else
      {
        //start sampling of ADC for next channel lastADCIndex
        //used or unused
        if(adcInterruptIndex<6)
        {
          ADCSRA |=B01000000;    // Start ADC Conversions  
        }
        else
        {
          adcInterruptIndex = 0;
        }
      }

     // PORTB &= ~0x1;
 }

// ------------------------- SHIFT registers and VU meter code ------------------------------------------


//
// Clear all LEDs (set to LOW)
//
void clearAllLeds()
{
  digitalWrite(SHIFT_LATCH_PIN, LOW);
  //shift out 5 bytes (40 bits-LEDs) with zero
  shiftOut(SHIFT_DATA_PIN, SHIFT_CLOCK_PIN, MSBFIRST, 0);
  shiftOut(SHIFT_DATA_PIN, SHIFT_CLOCK_PIN, MSBFIRST, 0);
  shiftOut(SHIFT_DATA_PIN, SHIFT_CLOCK_PIN, MSBFIRST, 0);
  shiftOut(SHIFT_DATA_PIN, SHIFT_CLOCK_PIN, MSBFIRST, 0);
  shiftOut(SHIFT_DATA_PIN, SHIFT_CLOCK_PIN, MSBFIRST, 0);  
  digitalWrite(SHIFT_LATCH_PIN, HIGH);
}
