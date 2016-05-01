
int okreni_90 (int broj, int polja){
    int resenje=0;int x[6];int y[6];
    
    for (int j=polja; j > 0; j--){x[j] = broj%10; broj /= 10;}   //rasclanjivanje broja na elemente
    for (int j=1; j<=polja; j++){y[polja-x[j]+1]=j;}             //preracun za okretanje za 90 stepeni

    for (int g=1; g<=polja; g++){resenje += y[g]*pow(10, g-1);} 
    return resenje;       
}
int okreni180 (int broj, int polja){
    int resenje=0;int x[6];int y[6];
    
    for (int j=polja; j > 0; j--){x[j] = broj%10; broj /= 10;}   //rasclanjivanje broja na elemente
    for(int j=1; j<=polja; j++){y[polja-x[j]+1]=j;}              //preracun za okretanje za 180 stepeni
    for(int j=1; j<=polja; j++){x[j]=y[polja-j+1];}
    
    for (int g=1; g<=polja; g++){resenje += x[g]*pow(10, g-1);} 
    return resenje;       
}

// Formirano je 4,096 work-group sa po 256 work-items (4,096*256=1,048,576 tj. 2na20)
// work-group velicina =256 (work-items). Definisano u clojure. (enq-nd! cqueue reduction-scalar (work-size [num-items] [workgroup-size (=256 mi smo definisali)]) nil profile-event)
__kernel void reduction_scalar(__global int* data,
                              __global int* brpolja,
                              __local int4* partial_sums,         //resenja
                              __local int* partial_podaci,        //lokalni podaci preneti iz globalne memorije
                              __local int* x,
                              __local int* y,                                                            
                              __global int4* output) {   
   
   int tr,j,g,f;        
   int gid = get_global_id(0);
   int lid = get_local_id(0);
   int gsize = get_local_size(0);
       
   partial_podaci[lid] = data[gid];
   int polja=brpolja[0];
   barrier(CLK_LOCAL_MEM_FENCE); 

                
   for (int k=0; k<=gsize/2; k++){        
        //j=polja; while (tr > 0){x[j] = tr%10; tr /= 10; j--;}      
        tr=partial_podaci[k]; for (int j=polja; j > 0; j--){x[j] = tr%10; tr /= 10;}    //rasclanjivanje broja na elemente
         
        int bre=1;int i=0;
        while (i < polja && bre==1){
            for(j=i+1; j<=polja; j++){
                if (!((x[i]==x[j]) || (abs(x[i]-x[j])==abs(i-j)))){
                    if (j == polja && i == polja-1) {                    //resenje
                       partial_sums[lid].s0 = partial_podaci[k];                       //tabla osnovna sa resenjem bez okretanja                                          
                       partial_sums[lid].s1 = okreni_90(partial_sums[lid].s0, polja);  //okrecemo tablu za 90 stepeni.  0->90
                       partial_sums[lid].s2 = okreni180(partial_sums[lid].s0, polja);  //okrecemo tablu za 180 stepeni. 0->180
                       partial_sums[lid].s3 = okreni180(partial_sums[lid].s1, polja);  //okrecemo tablu za 180 stepeni. 90->270                                                                                                                               
                    }                  
                }
                else {bre=0;}
            }
        i++;}
        barrier(CLK_LOCAL_MEM_FENCE);     
   } 
     
   if(lid == 0) {output[get_group_id(0)].s0 = partial_sums[lid].s0;
                 output[get_group_id(0)].s1 = partial_sums[lid].s1;
                 output[get_group_id(0)].s2 = partial_sums[lid].s2;
                 output[get_group_id(0)].s3 = partial_sums[lid].s3;}       //Vraca vrednost pojedinacne work-group (Na kraju krajeva za svih 4,096 komada)    

}
 
__kernel void reduction_scalar2(__global int* data,
                                __global int* brpolja,
                                __local int* partial_sums,          //resenja
                                __local int* partial_podaci,        //lokalni podaci preneti iz globalne memorije
                                __local int* x,                              
                                __global int* output) {   
   
   int trenutno,j;        
   int gid = get_global_id(0);
   int lid = get_local_id(0);
   int gsize = get_local_size(0);
       
   partial_podaci[lid] = data[gid];
   int polja=brpolja[0];
   barrier(CLK_LOCAL_MEM_FENCE); 

                
   for (int k=0; k<gsize; k++){
        trenutno=partial_podaci[k];   
        j=polja; while (trenutno > 0){x[j] = trenutno%10; trenutno /= 10; j--;}      //rasclanjivanje broja na elemente
    
        int bre=1;int i=0;
        while (i < polja && bre==1){
            for(j=i+1; j<=polja; j++){
                if (!((x[i]==x[j]) || (abs(x[i]-x[j])==abs(i-j)))){
                    if (j == polja && i == polja-1) {
                        partial_sums[lid] = partial_podaci[k];                 
                    }
                }
                else {bre=0;}
            }
        i++;}
        barrier(CLK_LOCAL_MEM_FENCE);     
   } 
    
   if(lid == 0) {output[get_group_id(0)] = partial_sums[lid];}        //Vraca vrednost pojedinacne work-group (Na kraju krajeva za svih 4,096 komada)
    
}
