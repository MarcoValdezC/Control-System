#include <Encoder.h>

int IN1  = 10; 
int IN2  = 11;
int PWM1 = 9;
int EN= 4;
int forWARDS  = 1; 
int backWARDS = 0;
const byte    encA     =  2;                  // Signal for channel A
const byte    encB     =  3;
//kp=9.0123585312173
//kd=3.87462795414925
double Setpoint, Input, Output;
const float pi = 3.14;
//double kp=7.039759626, ki=0, kd=1.068451969;
//double kp=6.8916108747095, ki=0.114454931035736, kd=4.59502748219678;
double kp=9.0123585312173, ki=0, kd=3.87462795414925;
//float Ts=466e-6;
long prevT = 0;
float outmax=20000.0;
float outmin=-20000.0;
float pwmmax=255;
float pwmmin=0.0;
float retro=0.0;
float error=0.0;
float error_ant=0.0;
float Integral=0.0;
float Derivada=0.0;
float u=0.0;
float uf=0.0;
float ui=0.0;
float resolucion=27300;

Encoder myEnc(encA,encB);
void setup() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(PWM1, LOW);
  analogWrite(EN, LOW);
  pinMode(EN,OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(PWM1, OUTPUT);
 double po1=13650;
 
  Setpoint = 13650;//13650/4;
  // put your setup code here, to run once:
Serial.begin(9600);
Serial.println("Basic Encoder Test:");
}
long oldPosition  = -999;
void loop() {
  // put your main code here, to run repeatedly:

  long newPosition = myEnc.read();
  if (newPosition != oldPosition) {
    oldPosition = newPosition;
    Serial.println((newPosition*2*pi)/resolucion);
  }
  Input=newPosition;
  error= Setpoint-newPosition;
  long currT = micros();
  
  float Ts = ((float)(currT-prevT))/1.0e6;
    prevT = currT;
  Integral= Integral+Ts*error;
if(Integral>outmax) Integral=outmax;
else if( Integral<outmin) Integral=outmin;
Derivada= (error-error_ant)/Ts;
u=kp*error+ki*Integral+kd*Derivada;

if(u>outmax)u=outmax;
else if(u<outmin)u=outmin;

error_ant=error;
//uf=((u-outmin)*((pwmmax-pwmmin)/(outmax-outmin)))+pwmmin;
uf=map(u,outmin,outmax,-255,pwmmax);
//Serial.println(uf);
//ui=(int)(uf+0.5);
//Serial.println("U");
//Serial.println(ui);
RunMotor(uf);
}

void RunMotor(double Usignal){  
  if (Setpoint-Input==0){
    shaftrev(IN1,IN2,PWM1,backWARDS, 0);
    digitalWrite(EN, LOW);
    //Serial.println("cero");
  }else if(Usignal>=0){
    
    if(Usignal<200 && error >= 140){
   Usignal=210; 
    }
    else if(error<5){
        digitalWrite(EN, LOW);
        }
    shaftrev(IN1,IN2,PWM1,backWARDS, Usignal);
    //Serial.println("positivo");
    
    //Serial.println(Usignal);
  }else{
    if(Usignal>-200 && error <= -140){
      Usignal=-210;
      }
     
      shaftrev(IN1,IN2,PWM1,forWARDS, -1*Usignal);
      //Serial.println("negativo");
      //Serial.println(-1*Usignal);
  }   
}

// Function that set DC_driver to move the motor
void shaftrev(int in1, int in2, int PWM, int sentido,int Wpulse){  
  if(sentido == 0){ //backWARDS
    digitalWrite(in2, HIGH);
    digitalWrite(in1, LOW);
    analogWrite(PWM,Wpulse);
    digitalWrite(EN, HIGH);
    
    }
  if(sentido == 1){ //forWARDS
    digitalWrite(in2, LOW);
    digitalWrite(in1, HIGH);
    analogWrite(PWM,Wpulse);    
    digitalWrite(EN, HIGH); 
    }
}
