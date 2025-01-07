
use candid::Principal;
use crate::AUTHORIZED_PRINCIPALS;


pub fn ensure_authorized(principal: Principal) {
    AUTHORIZED_PRINCIPALS.with(|principals| {
        let mut principals = principals.borrow_mut();
        if !principals.contains_key(&principal) {
            principals.insert(principal, true);
        }
    });
}


pub fn is_authenticated() -> Result<(), String> {
    let caller = ic_cdk::caller();
    AUTHORIZED_PRINCIPALS.with(|principals| {
        if principals.borrow().contains_key(&caller) {
            Ok(())
        } else {
            Err("Caller is not authorized".to_string())
        }
    })
}

#[ic_cdk::update(guard = "is_authenticated")]
fn add_authorized_principal(principal: Principal) {
    ensure_authorized(principal);
}

#[ic_cdk::query]
fn is_principal_authorized(principal: Principal) -> bool {
    AUTHORIZED_PRINCIPALS.with(|principals| {
        principals.borrow().contains_key(&principal)
    })
}


