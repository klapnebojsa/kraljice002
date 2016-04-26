
// Formirano je 4,096 work-group sa po 256 work-items (4,096*256=1,048,576 tj. 2na20)
// work-group velicina =256 (work-items). Definisano u clojure. (enq-nd! cqueue reduction-scalar (work-size [num-items] [workgroup-size (=256 mi smo definisali)]) nil profile-event)
__kernel void reduction_scalar(__global int* data,
                              __global int* brpolja,
                              __local int* partial_sums,
                              __local int* partial_podaci,
                              __local int* x,                              
                              __global int* output) {   
   
   int trenutno,j,s,bre;        
   int gid = get_global_id(0);
   int lid = get_local_id(0);
   int gsize = get_local_size(0);
       
   partial_podaci[lid] = data[gid];
   int polja=brpolja[0];
   barrier(CLK_LOCAL_MEM_FENCE); 

                
   for (int k=0; k<gsize; k++){
        trenutno=partial_podaci[k];   
        s=polja; while (trenutno > 0){x[s] = trenutno%10; trenutno /= 10; s--;}      //rasclanjivanje broja na elemente
    
        bre=1;int i=0;
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
    
   if(lid == 0) {
        output[get_group_id(0)] = partial_sums[lid];        //Vraca vrednost pojedinacne work-group (Na kraju krajeva za svih 4,096 komada)
   }
    
}


