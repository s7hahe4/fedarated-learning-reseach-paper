#include<iostream>
using namespace std;

#define d 256

void rabinkarp(string text,string pattern,int prime){

    int textsize=text.size();
    int psize=pattern.size();
    int texthash=0;
    int phash=0;
    bool match=false;
    int multi=1;
    int i,j;

    for(i=0;i<psize-1;i++){
        multi=(d*multi)%prime;
    }
    for(i=0;i<psize;i++){

        texthash=(d*texthash+text[i])%prime;
        phash=(d*phash+pattern[i])%prime;
    }

    for(i=0;i<=textsize-psize;i++){


        if(texthash==phash){

            match=true;

            for(j=0;j<psize;j++) {
                if(text[i+j]!=pattern[j]){

                    match=false;
                    break;

                }

            }
cout<<"the index is"<<i<<endl;
        }

        if(i<textsize-psize){

           texthash=(d*(texthash-text[i]*multi)+text[i+psize])%prime;

        }
        if(texthash<0){
            texthash=(texthash+prime);
        }

    }

}
int main (){


string text="ABCDEEFGTS";
string pattern="GTS";
int prime=101;

rabinkarp(text,pattern,prime);

}
