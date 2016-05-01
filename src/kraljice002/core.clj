(ns ch10-cisto.core
  (:require [midje.sweet :refer :all]
            [clojure.java.io :as io]
            [clojure.core.async :refer [chan <!!]]
            [uncomplicate.commons.core :refer [with-release]]            
            [uncomplicate.clojurecl
             [core :refer :all]
             [info :refer :all]
             [legacy :refer :all]
             [constants :refer :all]
             [toolbox :refer :all]
             [utils :refer :all]]
            [vertigo
             [bytes :refer [buffer direct-buffer byte-seq byte-count slice]]
             [structs :refer [int8 int32 int64 wrap-byte-seq]]])
  (:import [org.jocl CL]))


(set! *unchecked-math* true)
;(try 
 (with-release [;dev (nth  (sort-by-cl-version (devices (first (platforms)))) 0)
                platformsone (first (platforms))
                 
                dev (nth  (sort-by-cl-version (devices platformsone)) 0)
                ;dev (nth  (sort-by-cl-version (devices platformsone)) 1)
                ctx (context [dev])
                 cqueue (command-queue-1 ctx dev :profiling)
                ]
  
 (println (vendor platformsone))
 (println "dev: " (name-info dev))  
   

  (facts
   "Listing on page 225."
   (let [program-source
         (slurp (io/reader "examples/reduction.cl"))
         num-items (Math/pow 2 20)                    ;2 na 20-tu = 1048576
         bytesize (* num-items Float/BYTES)           ;Float/BYTES = 4    =>   bytesize = 4 * 2na20 = 4 * 1048576 = 4194304.0
         workgroup-size 256
         notifications (chan)
         follow (register notifications)
         brpolja (int-array 1 6)         
         data #_(int-array                  ;deo za formiranje ulaznih parametara za problem 8 dama
                  (with-local-vars [p ()]
                    (let [k (atom 0) n (aget brpolja 0) x (make-array Integer/TYPE n)]
                      (while (or (pos? @k) (zero? @k))
                        (aset x @k (inc (aget x @k)))
                        (if-not (> (aget x @k) n)
                          (do                            ;then 1
                            (if (= @k (- n 1))
                              (var-set p (conj @p (Integer. (clojure.string/join "" (vec x))))) ;then 2
                              (aset x (swap! k inc) 0)))                                        ;else 2
                          (swap! k dec))))               ;else 1
                    @p))
         (int-array (clojure.edn/read-string (slurp "podaci/kraljice.dat")))     ;deo za citanje ulaznih podataka za problem 8 dama
         
         cl-partial-sums (* workgroup-size Float/BYTES)       ;4 * 256 = 1024
         ;partial-output (int-array (/ bytesize workgroup-size))      ;4*2na20 / 256 = 4*2na20 / 2na8 = 4*2na12   - niz od 16384 elemenata
         partial-output (int-array (/ bytesize workgroup-size))        
         output (int-array 1)               ;pocetna vrednost jedan clan sa vrednoscu 0.0
         ]   
     (with-release [cl-data (cl-buffer ctx bytesize :read-only)
                    cl-brpolja (cl-buffer ctx bytesize :read-only)                                        
                    cl-output (cl-buffer ctx bytesize :write-only)
                    cl-partial-output (cl-buffer ctx (/ bytesize workgroup-size)   ;kreira cl_buffer objekat u kontekstu ctx velicine (4 * 2na20 / 256 = 2na14) i read-write ogranicenjima
                                                 :read-write)
                    cl-partial-podaci (* workgroup-size Float/BYTES)       ;4 * 256 = 1024
                    cl-x (* 6 workgroup-size Float/BYTES)
                    cl-y (* 6 workgroup-size Float/BYTES)                   
                    prog (build-program! (program-with-source ctx [program-source]))   ;kreira program u kontekstu ctx sa kodom programa u kojem se nalaze tri kernela 
                    ;naive-reduction (kernel prog "naive_reduction")            ;definise kernel iz prog
                    reduction-scalar (kernel prog "reduction_scalar")          ;definise kernel iz prog
                    reduction-scalar2 (kernel prog "reduction_scalar2")          ;definise kernel iz prog                    
                    ;reduction-vector (kernel prog "reduction_vector")          ;definise kernel iz prog
                    ;reduction-complete (kernel prog "reduction_complete")      ;definise kernel iz prog                    
                    profile-event (event)                  ;kreira novi cl_event (dogadjaj)
                    ;profile-event1 (event)                 ;          -||-
                    profile-event2 (event)                 ;          -||- 
                    ;profile-event3 (event)                 ;          -||-       
                    ]
       ;(println "(apply + (float-array (range 0" num-items "))): " (apply + data))

       (facts
        (println "============ Napredni algoritam ======================================")
        ;; ============= Napredni algoritam ====================================
        (set-args! reduction-scalar cl-data cl-brpolja cl-partial-sums cl-partial-podaci cl-x cl-y cl-output) => reduction-scalar
        ;(set-args! reduction-scalar cl-data cl-partial-sums cl-partial-output)  ;setovanje promenjivih u kernelu        reduction-scalar
                                                                                 ;cl-partial-sums=1024  i         
        ;=> reduction-scalar                                                      ;cl-partial-output = cl_buffer objekat u kontekstu ctx velicine (4 * 2na20 / 256 = 2na14) i read-write ogranicenjima
        
        (enq-write! cqueue cl-data data) => cqueue                                 ;SETUJE VREDNOST GLOBALNE PROMENJIVE cl-data SA VREDNOSCU data
        (enq-write! cqueue cl-brpolja brpolja) => cqueue        
        
        (enq-nd! cqueue reduction-scalar                       ;asinhrono izvrsava kernel u uredjaju. cqueue, kernel koji se izvrsava
                 (work-size [num-items]             ;[2na20]  sa ovim poljem vraca niz resenja (za svaki work-group posebno).
                                                    ;ako ovo izbrisemo vraca resenje samo u prvom elementu niza tj. konacan zbir svih work-group 
                            [256]                  ;[256] 
                            )       
                 nil profile-event)                            ;wait_event - da li da se ceka zavrsetak izvrsenja navedenih event-a tj proile-event1
        (follow profile-event)
        (enq-read! cqueue cl-output partial-output)
 
        ;(long (first partial-output)) => workgroup-size
        ;(println "partial-output POJEDINACNA RESENJA: " (seq partial-output))

       
        (finish! cqueue)
        (println "Scalar reduction time:"
                 (-> (<!! notifications) :event profiling-info durations :end))
       ;(println "UKUPAN ZBIR MEDJUSUMA____" (apply + (float-array (seq partial-output))))
       ;(println "UKUPAN ZBIR MEDJUSUMA____" (seq partial-output)) 
       (println "    RESENJA____: " (disj (set (seq partial-output)) 0))        
       ;(println "elemenata izlaza (" bytesize "=" num-items "*" Float/BYTES ") /" workgroup-size "=" (/ bytesize workgroup-size)) 

        )
       
       #_(facts
        (println "============ Najosnovniji algoritam ======================================")
        ;; ============= Najosnovniji algoritam ====================================
         (set-args! reduction-scalar2 cl-data cl-brpolja cl-partial-sums cl-partial-podaci cl-x cl-output) => reduction-scalar2

        (enq-write! cqueue cl-data data) => cqueue                                 ;SETUJE VREDNOST GLOBALNE PROMENJIVE cl-data SA VREDNOSCU data
         (enq-write! cqueue cl-brpolja brpolja) => cqueue        
        
        (enq-nd! cqueue reduction-scalar2                       ;asinhrono izvrsava kernel u uredjaju. cqueue, kernel koji se izvrsava
                  (work-size [num-items]             ;[2na20]  sa ovim poljem vraca niz resenja (za svaki work-group posebno).
                                                     ;ako ovo izbrisemo vraca resenje samo u prvom elementu niza tj. konacan zbir svih work-group 
                             [256]                  ;[256] 
                             )       
                 (events profile-event) profile-event2)                            ;wait_event - da li da se ceka zavrsetak izvrsenja navedenih event-a tj proile-event1
         (follow profile-event2)
        (enq-read! cqueue cl-output partial-output)
       
        (finish! cqueue)
        (println "Scalar reduction 2 time:"
                 (-> (<!! notifications) :event profiling-info durations :end))
       ;(println "UKUPAN ZBIR MEDJUSUMA____" (apply + (float-array (seq partial-output))))
       ;(println "UKUPAN ZBIR MEDJUSUMA____" (seq partial-output)) 
       (println "    RESENJA____: " (disj (set (seq partial-output)) 0))        
       ;(println "elemenata izlaza (" bytesize "=" num-items "*" Float/BYTES ") /" workgroup-size "=" (/ bytesize workgroup-size)) 
       

        (println "---------------KRAJ -------------------")   
        ;(println (seq data))
        )       
       
       ))))
  
 ;(catch Exception e (println "Greska 11111111: " (.getMessage e))))
